# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from datetime import datetime
import math

import accelerate
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils.deepspeed import (
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
)

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


from qwen_vl_utils import process_vision_info
from open_r1 import grpo_variants
from open_r1.sova import apply_sova_advantages, validate_sova_configuration

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class ProcessLogger:
    def __init__(self, prefix=""):
        self.pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            os.getenv("PRIVATE_DATA_ROOT"),
            os.getenv("WANDB_NAME"),
            "debug_logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}_pid{self.pid}.log")
        
    def log(self, message):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

token_logger = ProcessLogger("token")


@dataclass
class _NGRPORolloutCache:
    prompt_completion_ids: torch.Tensor
    prompt_inputs: dict[str, Any]
    prompt_length: int
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor
    advantages: torch.Tensor
    all_correct_groups: torch.Tensor
    rollout_metrics: dict[str, float]
    remaining_reuses: int
    batch_fingerprint: tuple


class _NGRPOSkipUpdateCallback(TrainerCallback):
    def __init__(self, trainer: "Qwen2VLGRPOTrainer") -> None:
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer.callback_handler.callbacks[-1] is not self:
            raise RuntimeError("NGRPO skip callback must run last")
        if not isinstance(kwargs.get("optimizer"), DeepSpeedOptimizerWrapper):
            raise RuntimeError("NGRPO skip control requires DeepSpeedOptimizerWrapper")
        if not isinstance(kwargs.get("lr_scheduler"), DeepSpeedSchedulerWrapper):
            raise RuntimeError("NGRPO skip control requires DeepSpeedSchedulerWrapper")
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._ngrpo_step_gate.before_counter_increment(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer._ngrpo_step_gate.after_counter_increment(state, control):
            self.trainer._ngrpo_skip_optimizer_step = False
        return control


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        loss_type: str = "grpo",
        use_sova: bool = False,
        sova_mode: str = "negative",
        sova_lambda_positive: float = 0.0,
        sova_lambda_negative: float = 0.0,
        sova_positive_virtual_total_reward: float = 1.5,
        sova_negative_virtual_total_reward: float = 2.0,
        ngrpo_num_iterations: int = grpo_variants.NGRPO_NUM_ITERATIONS,
        alpha: Optional[float] = 1.4,
        generate_temperature: Optional[float] = 1.0,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        loss_types = {"grpo", "tw_grpo", "ngrpo", "avspo"}
        if loss_type not in loss_types:
            allowed = ", ".join(sorted(loss_types))
            raise ValueError(f"loss_type must be one of: {allowed}")
        self.loss_type = loss_type
        self.ngrpo_num_iterations = int(ngrpo_num_iterations)
        self._ngrpo_rollout_cache = None
        self._ngrpo_discard_reuse_slots = 0
        self._ngrpo_discarded_fingerprint = None
        self._ngrpo_attempted_microstep = 0
        self._ngrpo_skip_optimizer_step = False
        self._ngrpo_step_gate = grpo_variants.NGRPOStepGate()
        if self.loss_type == "ngrpo":
            if self.ngrpo_num_iterations < 2:
                raise ValueError("NGRPO requires ngrpo_num_iterations >= 2")
            if args.gradient_accumulation_steps != 1:
                raise ValueError("NGRPO rollout reuse requires gradient_accumulation_steps=1")
        self.use_sova = use_sova
        self.sova_mode = sova_mode
        self.sova_lambda_positive = float(sova_lambda_positive)
        self.sova_lambda_negative = float(sova_lambda_negative)
        self.sova_positive_virtual_total_reward = float(
            sova_positive_virtual_total_reward
        )
        self.sova_negative_virtual_total_reward = float(
            sova_negative_virtual_total_reward
        )
        if self.use_sova:
            if loss_type != "tw_grpo":
                raise ValueError("SOVA requires loss_type=tw_grpo")
            validate_sova_configuration(
                self.sova_mode,
                self.sova_lambda_positive,
                self.sova_lambda_negative,
                self.sova_positive_virtual_total_reward,
                self.sova_negative_virtual_total_reward,
            )
        self.avspo_state = (
            grpo_variants.AVSPOState() if loss_type == "avspo" else None
        )

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        self.vision_modules_keywords = ["visual"]
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False


        self.beta = args.beta
        if self.loss_type == "ngrpo" and self.beta != 0.0:
            raise ValueError("NGRPO requires beta=0")
        # Reference model
        if is_deepspeed_zero3_enabled():
            if self.beta > 0:
                if "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = None
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        self.sova_accuracy_reward_index = None
        self.sova_format_reward_index = None
        reward_func_names = [
            reward_func.config._name_or_path.split("/")[-1]
            if isinstance(reward_func, PreTrainedModel)
            else reward_func.__name__
            for reward_func in self.reward_funcs
        ]
        if self.use_sova:
            if reward_func_names != ["accuracy_reward", "format_reward"]:
                raise ValueError(
                    "SOVA requires reward_funcs in the order: accuracy, format"
                )
            self.sova_accuracy_reward_index = 0
            self.sova_format_reward_index = 1
        if loss_type in {"ngrpo", "avspo"} and reward_func_names != [
            "origin_accuracy_reward"
        ]:
            raise ValueError(
                f"{loss_type.upper()} requires the binary origin_accuracy reward"
            )

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        if self.use_sova and self.num_generations != 8:
            raise ValueError("SOVA requires the frozen TW-GRPO num_generations=8")
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=generate_temperature,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.alpha = alpha
        if loss_type == "ngrpo":
            self.epsilon_low = grpo_variants.NGRPO_EPSILON_NEGATIVE
            self.epsilon_high = grpo_variants.NGRPO_EPSILON_POSITIVE
        elif loss_type == "avspo":
            self.epsilon_low = grpo_variants.AVSPO_CLIP_EPSILON
            self.epsilon_high = grpo_variants.AVSPO_CLIP_EPSILON
        else:
            self.epsilon_low = 0.20
            self.epsilon_high = 0.28

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.loss_type == "ngrpo":
            if version.parse(transformers.__version__) != version.parse("4.50.0"):
                raise RuntimeError("NGRPO transactional skipping requires transformers==4.50.0")
            if version.parse(accelerate.__version__) != version.parse("1.2.1"):
                raise RuntimeError("NGRPO transactional skipping requires accelerate==1.2.1")
            if not self.is_deepspeed_enabled:
                raise RuntimeError("NGRPO transactional skipping requires DeepSpeed")
            if isinstance(self.train_dataset, IterableDataset):
                raise RuntimeError("NGRPO rollout reuse requires a sized training dataset")
            # Rank-local rollout reuse relies on accelerate assigning micro-batch c to
            # rank c % num_processes. Both options below break that mapping.
            accelerator_config = self.args.accelerator_config
            if getattr(accelerator_config, "split_batches", False):
                raise RuntimeError(
                    "NGRPO rollout reuse requires accelerator_config.split_batches=False"
                )
            if getattr(accelerator_config, "dispatch_batches", None) not in (None, False):
                raise RuntimeError(
                    "NGRPO rollout reuse requires accelerator_config.dispatch_batches to be unset or False"
                )
            self.add_callback(_NGRPOSkipUpdateCallback(self))

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        if self.loss_type == "ngrpo":
            requested = resume_from_checkpoint
            if requested is None:
                requested = self.args.resume_from_checkpoint
            if requested not in (None, False):
                raise RuntimeError(
                    "NGRPO cannot resume from a checkpoint: the rollout-phase counter is "
                    "not checkpointed, and skipped updates decouple global_step from the "
                    "dataloader position, so the reuse cycle would restart out of phase"
                )
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def _get_train_sampler(self):
        if self.loss_type != "ngrpo":
            return super()._get_train_sampler()
        global_batch_size = (
            self._train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return grpo_variants.RepeatGlobalBatchSampler(
            self.train_dataset,
            batch_size=global_batch_size,
            repeat_count=self.ngrpo_num_iterations,
            seed=seed,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.loss_type != "ngrpo":
            return super().training_step(model, inputs, num_items_in_batch)
        if self._ngrpo_step_gate.armed or self._ngrpo_skip_optimizer_step:
            raise RuntimeError("NGRPO skip state leaked across training steps")

        entry_global_step = self.state.global_step
        original_backward = self.accelerator.backward

        def guarded_backward(loss, **kwargs):
            grpo_variants.ngrpo_backward_or_arm(
                original_backward,
                loss,
                self._ngrpo_skip_optimizer_step,
                self._ngrpo_step_gate,
                entry_global_step,
                **kwargs,
            )

        self.accelerator.backward = guarded_backward
        completed = False
        try:
            result = super().training_step(model, inputs, num_items_in_batch)
            if self._ngrpo_skip_optimizer_step and not self._ngrpo_step_gate.armed:
                raise RuntimeError("NGRPO skip did not intercept backward")
            completed = True
            return result
        finally:
            self.accelerator.backward = original_backward
            if completed:
                self._ngrpo_attempted_microstep += 1

    @staticmethod
    def _get_per_token_logps(model, input_ids, return_full_logps=True, **kwargs):
        logits = model(input_ids, **kwargs).logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        per_logps = [] if return_full_logps else None
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
            if return_full_logps:
                per_logps.append(log_probs)
        return (
            torch.stack(per_token_logps),
            torch.stack(per_logps) if return_full_logps else None,
        )

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    @staticmethod
    def _ngrpo_batch_fingerprint(inputs) -> tuple:
        """Identify a rollout batch so reuse cannot silently target other data."""
        fingerprint = []
        for example in inputs:
            identity = tuple(
                f"{key}={example[key]!r}"
                for key in ("problem", "solution", "video", "image", "problem_type")
                if isinstance(example.get(key), (str, int, float))
            )
            fingerprint.append(identity or (repr(example.get("prompt")),))
        return tuple(fingerprint)

    def _record_ngrpo_policy_metrics(
        self,
        ratios: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        all_correct_groups: torch.Tensor,
        iteration: int,
    ) -> None:
        self._metrics["ngrpo/policy_iteration"].append(float(iteration + 1))
        if iteration == 0:
            # The rollout iteration reuses its own forward pass as the reference
            # policy, so every ratio is exactly one and no clip can bind. Recording
            # it would halve the reported clip statistics under any smoothing.
            return
        keep_sequences = (~all_correct_groups).repeat_interleave(
            self.num_generations
        )
        mask = completion_mask.bool() & keep_sequences.unsqueeze(1)
        expanded_advantages = advantages.unsqueeze(1).expand_as(ratios)
        clipped = (
            ((expanded_advantages >= 0) & (ratios > 1.0 + self.epsilon_high))
            | ((expanded_advantages < 0) & (ratios < 1.0 - self.epsilon_low))
        ) & mask
        local = torch.stack(
            (
                (ratios.float().sub(1.0).abs() * mask).sum(),
                clipped.float().sum(),
                mask.float().sum(),
            )
        )
        global_stats = self.accelerator.reduce(local, reduction="sum")
        count = max(global_stats[2].item(), 1.0)
        self._metrics["ngrpo/ratio_abs_deviation"].append(
            global_stats[0].item() / count
        )
        self._metrics["ngrpo/clip_fraction"].append(
            global_stats[1].item() / count
        )

    def _compute_reused_ngrpo_loss(self, model, phase: int) -> torch.Tensor:
        cache = self._ngrpo_rollout_cache
        if cache is None or cache.remaining_reuses < 1:
            raise RuntimeError("NGRPO rollout cache is not reusable")

        per_token_logps, _ = self._get_per_token_logps(
            model,
            cache.prompt_completion_ids,
            return_full_logps=False,
            **cache.prompt_inputs,
        )
        per_token_logps = per_token_logps[:, cache.prompt_length - 1 :]
        if per_token_logps.shape != cache.old_per_token_logps.shape:
            raise RuntimeError("NGRPO current and old log-probability shapes differ")

        ratios = torch.exp(per_token_logps - cache.old_per_token_logps)
        surrogate = grpo_variants.clipped_surrogate(
            ratios,
            cache.advantages.unsqueeze(1),
            self.epsilon_low,
            self.epsilon_high,
        )
        per_token_loss = -surrogate
        sequence_loss = (
            (per_token_loss * cache.completion_mask).sum(dim=1)
            / cache.completion_mask.sum(dim=1)
        )
        filtered = grpo_variants.ngrpo_filtered_sequence_loss(
            sequence_loss,
            cache.all_correct_groups,
            self.num_generations,
            self.accelerator.num_processes,
            self.accelerator.reduce,
        )
        if filtered.skip_update:
            raise RuntimeError("A cached NGRPO rollout changed its keep decision")
        self._ngrpo_skip_optimizer_step = False

        for key, value in cache.rollout_metrics.items():
            self._metrics[key].append(value)
        self._record_ngrpo_policy_metrics(
            ratios,
            cache.advantages,
            cache.completion_mask,
            cache.all_correct_groups,
            phase,
        )
        self._metrics["ngrpo/rollout_generated"].append(0.0)
        self._metrics["ngrpo/rollout_reused"].append(1.0)
        self._metrics["ngrpo/discarded_all_correct"].append(0.0)
        self._metrics["ngrpo/discarded_reuse_slot"].append(0.0)
        self._metrics["ngrpo/skipped_update"].append(0.0)
        self._metrics["ngrpo/successful_update"].append(1.0)
        self._metrics["kl"].append(0.0)

        cache.remaining_reuses -= 1
        if cache.remaining_reuses == 0:
            self._ngrpo_rollout_cache = None
        return filtered.loss

    @staticmethod
    def _detach_rollout_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.detach()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for key, value in inputs.items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for GRPO.
        Args:
            model: The model to train
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (new parameter)
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        if self.loss_type == "ngrpo" and model.training:
            phase = self._ngrpo_attempted_microstep % self.ngrpo_num_iterations
            batch_fingerprint = self._ngrpo_batch_fingerprint(inputs)
            if phase > 0:
                if self._ngrpo_rollout_cache is not None:
                    if batch_fingerprint != self._ngrpo_rollout_cache.batch_fingerprint:
                        raise RuntimeError(
                            "NGRPO reuse received a batch that differs from the cached "
                            "rollout; the dataloader is not repeating global batches per rank"
                        )
                    return self._compute_reused_ngrpo_loss(model, phase)
                if self._ngrpo_discard_reuse_slots > 0:
                    if batch_fingerprint != self._ngrpo_discarded_fingerprint:
                        raise RuntimeError(
                            "NGRPO discard received a batch that differs from the dropped "
                            "rollout; the dataloader is not repeating global batches per rank"
                        )
                    self._ngrpo_discard_reuse_slots -= 1
                    self._ngrpo_skip_optimizer_step = True
                    self._metrics["ngrpo/policy_iteration"].append(float(phase + 1))
                    self._metrics["ngrpo/rollout_generated"].append(0.0)
                    self._metrics["ngrpo/rollout_reused"].append(0.0)
                    self._metrics["ngrpo/discarded_all_correct"].append(0.0)
                    self._metrics["ngrpo/discarded_reuse_slot"].append(1.0)
                    self._metrics["ngrpo/skipped_update"].append(1.0)
                    self._metrics["ngrpo/successful_update"].append(0.0)
                    self._metrics["kl"].append(0.0)
                    return torch.zeros((), device=device)
                raise RuntimeError("NGRPO rollout cache is missing during reuse")
            if self._ngrpo_rollout_cache is not None:
                raise RuntimeError("NGRPO rollout cache survived a complete reuse cycle")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        if "image" in inputs[0]:
            images = [x["image"] for x in inputs]
        elif "video" in inputs[0]:
            videos = [x["video"] for x in inputs]
            video_inputs = []
            for (inp_idx, inp) in enumerate(inputs):
                new_inp = inp.copy()
                new_inp['prompt'][0]['content'][0]['text'] = inputs[inp_idx]["video"]
                video_path = inputs[inp_idx]["video"]
                video_inputs.append(process_vision_info(new_inp["prompt"])[0])

        if "image" in inputs[0] or "video" in inputs[0]:
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=images if "image" in inputs[0] else None,
                videos=video_inputs if "video" in inputs[0] else None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)


        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # import pdb; pdb.set_trace()

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        # Okay I am assuming that the inputs are Qwen2VL processor
        # and no video for now, repeat the image for each completion
        if "image" in inputs[0]:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        
        # XXX if input video
        # image_grid_thw is from image_process_qwen2_vl
        # https://github.com/huggingface/transformers/blob/dd16acb8a3e93b643aa374c9fb80749f5235c1a6/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L414
        # automatic process
        if "video" in inputs[0]:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if "second_per_grid_ts" in prompt_inputs:
                prompt_inputs["second_per_grid_ts"] = prompt_inputs["second_per_grid_ts"] * len(prompt_completion_ids)

        per_token_logps, per_logps = self._get_per_token_logps(
            model,
            prompt_completion_ids,
            return_full_logps=self.loss_type != "ngrpo",
            **prompt_inputs,
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        if per_logps is not None:
            per_logps = per_logps[:, prompt_length - 1 :]

        if self.beta > 0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, ref_per_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, **prompt_inputs
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, ref_per_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, **prompt_inputs
                        )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        else:
            ref_per_token_logps = per_token_logps.detach()
            ref_per_logps = per_logps.detach() if per_logps is not None else None
            
        # Following previous works like R1-V and TRL, we simplify the clipping mechanism which has been shown to work well in practice.
        # For reference, please see:
        # https://github.com/huggingface/trl/issues/2608#issuecomment-2609844003
        old_per_token_logps = per_token_logps.detach()
        if self.loss_type == "ngrpo" and model.training:
            old_per_token_logps = old_per_token_logps.clone()

        # Compute the KL divergence between the model and the reference model
        if self.beta > 0:
            diff = ref_per_token_logps - per_token_logps
            diff = torch.clamp(diff, min=-11.0, max=11.0) 
        else:
            diff = torch.zeros_like(per_token_logps)

        per_token_kl = torch.exp(diff) - (diff) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]): # true
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Add trainer to reward_kwargs
                reward_kwargs = {
                    key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # Pass per_token_logps, trainer, completion_ids and tag_tokens to the reward function
                reward_kwargs["per_token_logps"] = per_token_logps.detach()
                reward_kwargs["per_logps"] = (
                    per_logps.detach() if per_logps is not None else None
                )
                reward_kwargs["trainer"] = self
                reward_kwargs["completion_ids"] = completion_ids
                reward_kwargs["tag_tokens"] = getattr(self, 'tag_tokens', None)
                try:
                    reward_kwargs["video_path"] = video_path
                except NameError:
                    reward_kwargs["video_path"] = None
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Baseline group-normalized advantages.
        grouped_rewards = rewards.view(-1, self.num_generations)
        raw_mean_grouped_rewards = grouped_rewards.mean(dim=1)
        raw_std_grouped_rewards = grouped_rewards.std(dim=1)
        mean_grouped_rewards = raw_mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = raw_std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        baseline_advantages = (rewards - mean_grouped_rewards) / (
            std_grouped_rewards + 1e-4
        )
        advantages = baseline_advantages
        ngrpo_result = None
        avspo_result = None
        avspo_acr = None
        avspo_threshold = None
        if self.loss_type == "ngrpo":
            ngrpo_result = grpo_variants.compute_ngrpo_advantages(
                rewards=rewards,
                num_generations=self.num_generations,
            )
            advantages = ngrpo_result.advantages
        elif self.loss_type == "avspo":
            collapsed = grpo_variants.find_avspo_collapsed_groups(
                rewards,
                self.num_generations,
            )
            avspo_acr, global_mean_reward = (
                grpo_variants.reduce_avspo_batch_statistics(
                    collapsed,
                    rewards,
                    self.accelerator.reduce,
                )
            )
            avspo_threshold = self.avspo_state.threshold
            avspo_result = grpo_variants.apply_avspo_advantages(
                baseline_advantages=baseline_advantages,
                rewards=rewards,
                num_generations=self.num_generations,
                global_acr=avspo_acr,
                adaptive_threshold=avspo_threshold,
            )
            advantages = avspo_result.advantages
            if model.training:
                self.avspo_state = grpo_variants.advance_avspo_state(
                    self.avspo_state,
                    global_acr=avspo_acr,
                    mean_reward=global_mean_reward,
                )
        sova_result = None
        if self.use_sova:
            sova_result = apply_sova_advantages(
                baseline_advantages=baseline_advantages,
                total_rewards=rewards,
                accuracy_rewards=rewards_per_func[:, self.sova_accuracy_reward_index],
                format_rewards=rewards_per_func[:, self.sova_format_reward_index],
                num_generations=self.num_generations,
                mode=self.sova_mode,
                lambda_positive=self.sova_lambda_positive,
                lambda_negative=self.sova_lambda_negative,
                positive_virtual_total_reward=self.sova_positive_virtual_total_reward,
                negative_virtual_total_reward=self.sova_negative_virtual_total_reward,
            )
            advantages = sova_result.advantages

        # # x - x.detach() allows for preserving gradients from x
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl) # default 0.04

        ratios = torch.exp(per_token_logps - old_per_token_logps)
        surrogate = grpo_variants.clipped_surrogate(
            ratios,
            advantages.unsqueeze(1),
            self.epsilon_low,
            self.epsilon_high,
        )
        per_token_loss = -(surrogate - self.beta * per_token_kl)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        if self.loss_type in {"grpo", "ngrpo", "avspo"}:
            # Original: Normalize each sequence first, then average
            sequence_loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            )
            if self.loss_type == "ngrpo":
                ngrpo_filtered = grpo_variants.ngrpo_filtered_sequence_loss(
                    sequence_loss,
                    ngrpo_result.all_correct_groups,
                    self.num_generations,
                    self.accelerator.num_processes,
                    self.accelerator.reduce,
                )
                loss = ngrpo_filtered.loss
                self._ngrpo_skip_optimizer_step = (
                    ngrpo_filtered.skip_update if model.training else False
                )
            else:
                loss = sequence_loss.mean()
        elif self.loss_type == "tw_grpo":
            # Focus on reasoning
            token_weights = compute_token_importance_kl_logs_uniform(per_logps, completion_mask, completion_ids, self.num_generations, max_weight=self.alpha)
            weighted_loss = per_token_loss * token_weights
            loss = (weighted_loss * completion_mask).sum() / completion_mask.sum()
        
        # import pdb; pdb.set_trace()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        self._metrics["advantages"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
        
        self._metrics["reward_mean"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if self.loss_type == "ngrpo":
            self._metrics["ngrpo/all_correct_filter_rate"].append(
                self.accelerator.gather_for_metrics(
                    ngrpo_result.all_correct_groups.float()
                ).mean().item()
            )
            self._metrics["ngrpo/effective_reward_mean"].append(
                self.accelerator.gather_for_metrics(
                    ngrpo_result.effective_mean
                ).mean().item()
            )
            self._metrics["ngrpo/effective_reward_std"].append(
                self.accelerator.gather_for_metrics(
                    ngrpo_result.effective_std
                ).mean().item()
            )
            self._record_ngrpo_policy_metrics(
                ratios,
                advantages,
                completion_mask,
                ngrpo_result.all_correct_groups,
                iteration=0,
            )
            self._metrics["ngrpo/rollout_generated"].append(1.0)
            self._metrics["ngrpo/rollout_reused"].append(0.0)
            self._metrics["ngrpo/discarded_all_correct"].append(
                float(ngrpo_filtered.skip_update)
            )
            self._metrics["ngrpo/discarded_reuse_slot"].append(0.0)
            self._metrics["ngrpo/skipped_update"].append(
                float(ngrpo_filtered.skip_update)
            )
            self._metrics["ngrpo/successful_update"].append(
                float(not ngrpo_filtered.skip_update)
            )

        if self.loss_type == "avspo":
            self._metrics["avspo/acr"].append(avspo_acr)
            self._metrics["avspo/threshold"].append(avspo_threshold)
            self._metrics["avspo/next_threshold"].append(
                self.avspo_state.threshold
            )
            self._metrics["avspo/virtual_count"].append(
                float(avspo_result.virtual_count)
            )
            self._metrics["avspo/activation_rate"].append(
                self.accelerator.gather_for_metrics(
                    avspo_result.active_groups.float()
                ).mean().item()
            )

        if self.use_sova:
            self._metrics["sova/positive_activation_rate"].append(
                self.accelerator.gather_for_metrics(
                    sova_result.positive_active_groups.float()
                ).mean().item()
            )
            self._metrics["sova/negative_activation_rate"].append(
                self.accelerator.gather_for_metrics(
                    sova_result.negative_active_groups.float()
                ).mean().item()
            )
            self._metrics["sova/lambda_positive"].append(
                self.sova_lambda_positive
            )
            self._metrics["sova/lambda_negative"].append(
                self.sova_lambda_negative
            )
            self._metrics["sova/positive_residual_mean"].append(
                self.accelerator.gather_for_metrics(
                    sova_result.positive_residual
                ).mean().item()
            )
            self._metrics["sova/negative_residual_mean"].append(
                self.accelerator.gather_for_metrics(
                    sova_result.negative_residual
                ).mean().item()
            )
            self._metrics["sova/scaled_positive_residual_mean"].append(
                self.accelerator.gather_for_metrics(
                    self.sova_lambda_positive * sova_result.positive_residual
                ).mean().item()
            )
            self._metrics["sova/scaled_negative_residual_mean"].append(
                self.accelerator.gather_for_metrics(
                    self.sova_lambda_negative * sova_result.negative_residual
                ).mean().item()
            )

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.loss_type == "ngrpo" and model.training:
            if ngrpo_filtered.skip_update:
                self._ngrpo_rollout_cache = None
                self._ngrpo_discard_reuse_slots = self.ngrpo_num_iterations - 1
                self._ngrpo_discarded_fingerprint = batch_fingerprint
            else:
                dynamic_metrics = {
                    "kl",
                    "ngrpo/policy_iteration",
                    "ngrpo/ratio_abs_deviation",
                    "ngrpo/clip_fraction",
                    "ngrpo/rollout_generated",
                    "ngrpo/rollout_reused",
                    "ngrpo/discarded_all_correct",
                    "ngrpo/discarded_reuse_slot",
                    "ngrpo/skipped_update",
                    "ngrpo/successful_update",
                }
                rollout_metrics = {
                    key: values[-1]
                    for key, values in self._metrics.items()
                    if values and key not in dynamic_metrics
                }
                self._ngrpo_rollout_cache = _NGRPORolloutCache(
                    prompt_completion_ids=prompt_completion_ids.detach(),
                    prompt_inputs=self._detach_rollout_inputs(prompt_inputs),
                    prompt_length=prompt_length,
                    completion_mask=completion_mask.detach(),
                    old_per_token_logps=old_per_token_logps.detach(),
                    advantages=advantages.detach(),
                    all_correct_groups=ngrpo_result.all_correct_groups.detach(),
                    rollout_metrics=rollout_metrics,
                    remaining_reuses=self.ngrpo_num_iterations - 1,
                    batch_fingerprint=batch_fingerprint,
                )
                self._ngrpo_discarded_fingerprint = None

        # import pdb; pdb.set_trace()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Flush accumulated GRPO-variant metrics through the Trainer logger."""
        metrics = {
            key: sum(values) / len(values)
            for key, values in self._metrics.items()
            if values
        }
        first_log_key = next(iter(logs), "")
        if first_log_key.startswith("eval_"):
            metrics = {f"eval_{key}": value for key, value in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

def compute_token_importance_kl_logs_uniform(per_logps, completion_mask, completion_ids, num_generations, max_weight=1.5):
    # Basic dimension checks
    if not isinstance(per_logps, torch.Tensor) or per_logps.ndim != 3:
        return torch.ones_like(per_logps[..., 0])
        
    total_size, token_length, vocab_size = per_logps.size()
    if total_size % num_generations != 0:
        return torch.ones_like(per_logps[..., 0])
        
    batch_size = total_size // num_generations
    
    # Reshape tensors for group-wise computation
    grouped_logps = per_logps.view(batch_size, num_generations, token_length, vocab_size)
    grouped_masks = completion_mask.view(batch_size, num_generations, token_length)
    grouped_masks = grouped_masks.unsqueeze(-1)  # (batch_size, num_generations, token_length, 1)
    
    # Create uniform distribution for masked positions (log space)
    uniform_logps = torch.full_like(grouped_logps, -math.log(vocab_size))
    
    # Set logps to uniform distribution for positions beyond sequence length
    masked_logps = torch.where(grouped_masks == 1, grouped_logps, uniform_logps)
    
    # Calculate mean distribution for each token position
    mean_logps = masked_logps.mean(dim=1, keepdim=True)  # (batch_size, 1, token_length, vocab_size)
    
    # Calculate KL divergence between each sequence and the mean
    diff = mean_logps - masked_logps
    diff = torch.clamp(diff, min=-11.0, max=11.0)
    token_kl = (torch.exp(diff) - diff - 1).sum(dim=-1)
    token_kl = token_kl.mean(dim=1)
    
    # Apply min-max normalization to KL divergence
    kl_min = token_kl.min(dim=1, keepdim=True)[0]
    kl_max = token_kl.max(dim=1, keepdim=True)[0]
    normalized_kl = (token_kl - kl_min) / (kl_max - kl_min + 1e-8)
    
    # Map normalized KL to weights range [1.0, max_weight]
    token_weights = 1.0 + (max_weight - 1.0) * normalized_kl
    # Repeat weights for each generation
    token_weights = token_weights.repeat_interleave(num_generations, dim=0)
    
    # Apply the completion mask to ensure consistency
    token_weights = token_weights * completion_mask

    # log token_weights
    # for per_logps, completion_id in zip(per_logps, completion_ids):
    #     # token_logger.log(f"per_token_logp: {per_logps}")
    #     token_logger.log(f"completion_id: {completion_id}")
    # token_logger.log(f"token_weights: {normalized_kl}")
    
    return token_weights
