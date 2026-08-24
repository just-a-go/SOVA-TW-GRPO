import os
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("PRIVATE_DATA_ROOT", tempfile.gettempdir())
os.environ.setdefault("WANDB_NAME", "codex-ngrpo-tests")

TRAINER_DEPS_AVAILABLE = all(
    find_spec(name) is not None
    for name in ("accelerate", "datasets", "qwen_vl_utils", "transformers", "trl")
)

if TRAINER_DEPS_AVAILABLE:
    from accelerate import Accelerator
    from accelerate.data_loader import BatchSamplerShard
    from accelerate.utils.deepspeed import (
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
    )
    from open_r1 import grpo_variants
    import open_r1.trainer.grpo_trainer as trainer_module
    from open_r1.trainer.grpo_trainer import (
        Qwen2VLGRPOTrainer,
        _NGRPORolloutCache,
        _NGRPOSkipUpdateCallback,
    )
    from transformers import Trainer
    from torch.utils.data import BatchSampler
else:
    Qwen2VLGRPOTrainer = None


@unittest.skipIf(Qwen2VLGRPOTrainer is None, "trainer dependencies are unavailable")
class NGRPOTrainerControlTest(unittest.TestCase):
    def test_skip_callback_accepts_exact_deepspeed_wrappers(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        optimizer_wrapper = DeepSpeedOptimizerWrapper(optimizer)
        scheduler_wrapper = DeepSpeedSchedulerWrapper(scheduler, [optimizer_wrapper])

        trainer = SimpleNamespace()
        callback = _NGRPOSkipUpdateCallback(trainer)
        trainer.callback_handler = SimpleNamespace(callbacks=[callback])
        control = SimpleNamespace()
        returned = callback.on_train_begin(
            None,
            SimpleNamespace(),
            control,
            optimizer=optimizer_wrapper,
            lr_scheduler=scheduler_wrapper,
        )
        self.assertIs(returned, control)

    def test_discarded_reuse_slot_skips_without_generating(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer.ngrpo_num_iterations = 2
        trainer._ngrpo_attempted_microstep = 1
        trainer._ngrpo_rollout_cache = None
        trainer._ngrpo_discard_reuse_slots = 1
        trainer._ngrpo_discarded_fingerprint = ()
        trainer._ngrpo_skip_optimizer_step = False
        trainer._metrics = defaultdict(list)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model = SimpleNamespace(training=True)

        loss = trainer.compute_loss(model, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(trainer._ngrpo_skip_optimizer_step)
        self.assertEqual(trainer._ngrpo_discard_reuse_slots, 0)
        self.assertEqual(trainer._metrics["ngrpo/discarded_reuse_slot"], [1.0])
        self.assertEqual(trainer._metrics["ngrpo/successful_update"], [0.0])
        self.assertNotIn("ngrpo/ratio_abs_deviation", trainer._metrics)
        self.assertNotIn("ngrpo/clip_fraction", trainer._metrics)

    def test_discarded_reuse_slot_rejects_a_different_batch(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer.ngrpo_num_iterations = 2
        trainer._ngrpo_attempted_microstep = 1
        trainer._ngrpo_rollout_cache = None
        trainer._ngrpo_discard_reuse_slots = 1
        trainer._ngrpo_discarded_fingerprint = ()
        trainer._ngrpo_skip_optimizer_step = False
        trainer._metrics = defaultdict(list)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model = SimpleNamespace(training=True)

        with self.assertRaisesRegex(RuntimeError, "differs from the dropped"):
            trainer.compute_loss(model, [{"problem": "other", "solution": "b"}])

    def test_reuse_rejects_a_different_batch(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer.ngrpo_num_iterations = 2
        trainer._ngrpo_attempted_microstep = 1
        trainer._ngrpo_rollout_cache = SimpleNamespace(batch_fingerprint=())
        trainer._ngrpo_discard_reuse_slots = 0
        trainer._ngrpo_skip_optimizer_step = False
        trainer._metrics = defaultdict(list)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model = SimpleNamespace(training=True)

        with self.assertRaisesRegex(RuntimeError, "differs from the cached"):
            trainer.compute_loss(model, [{"problem": "other", "solution": "b"}])

    def test_rollout_iteration_records_no_clip_statistics(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.num_generations = 2
        trainer.epsilon_low = 0.16
        trainer.epsilon_high = 0.24
        trainer._metrics = defaultdict(list)
        trainer.accelerator = SimpleNamespace(
            reduce=lambda value, reduction="sum": value
        )
        ratios = torch.ones((2, 3))

        trainer._record_ngrpo_policy_metrics(
            ratios,
            torch.tensor([1.0, -1.0]),
            torch.ones((2, 3), dtype=torch.int),
            torch.tensor([False]),
            iteration=0,
        )
        self.assertEqual(trainer._metrics["ngrpo/policy_iteration"], [1.0])
        self.assertNotIn("ngrpo/ratio_abs_deviation", trainer._metrics)
        self.assertNotIn("ngrpo/clip_fraction", trainer._metrics)

        trainer._record_ngrpo_policy_metrics(
            ratios,
            torch.tensor([1.0, -1.0]),
            torch.ones((2, 3), dtype=torch.int),
            torch.tensor([False]),
            iteration=1,
        )
        self.assertEqual(trainer._metrics["ngrpo/policy_iteration"], [1.0, 2.0])
        self.assertEqual(trainer._metrics["ngrpo/ratio_abs_deviation"], [0.0])
        self.assertEqual(trainer._metrics["ngrpo/clip_fraction"], [0.0])

    def test_train_refuses_to_resume_from_a_checkpoint(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer.args = SimpleNamespace(resume_from_checkpoint=None)

        with self.assertRaisesRegex(RuntimeError, "cannot resume from a checkpoint"):
            trainer.train(resume_from_checkpoint="outputs/checkpoint-100")

        trainer.args = SimpleNamespace(
            resume_from_checkpoint="outputs/checkpoint-100"
        )
        with self.assertRaisesRegex(RuntimeError, "cannot resume from a checkpoint"):
            trainer.train()

    def test_fresh_rollout_builds_cache_then_reuses_without_generation(self):
        class Accelerator:
            device = torch.device("cpu")
            num_processes = 1

            @staticmethod
            def reduce(value, reduction="sum"):
                return value

            @staticmethod
            def gather_for_metrics(value):
                return value

        class Processor:
            eos_token_id = 2

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.zeros((len(text), 1), dtype=torch.long),
                    "attention_mask": torch.ones((len(text), 1), dtype=torch.long),
                }

            @staticmethod
            def batch_decode(ids, skip_special_tokens=True):
                return ["answer" for _ in range(ids.size(0))]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = torch.nn.Parameter(torch.tensor(0.0))
                self.generate_calls = 0

            def generate(self, input_ids, generation_config, **kwargs):
                self.generate_calls += 1
                return torch.tensor([[0, 1, 2], [0, 1, 2]])

            def forward(self, input_ids, **kwargs):
                batch, length = input_ids.shape
                logits = torch.zeros(batch, length, 3)
                logits[..., 0] = self.theta
                return SimpleNamespace(logits=logits)

        def reward(prompts, completions, **kwargs):
            return [1.0, 0.0]

        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.accelerator = Accelerator()
        trainer.processing_class = Processor()
        trainer.max_prompt_length = None
        trainer.generation_config = SimpleNamespace(num_return_sequences=2)
        trainer.num_generations = 2
        trainer.reward_funcs = [reward]
        trainer.reward_processing_classes = [None]
        trainer.beta = 0.0
        trainer.loss_type = "ngrpo"
        trainer.epsilon_low = 0.16
        trainer.epsilon_high = 0.24
        trainer.use_sova = False
        trainer.ngrpo_num_iterations = 2
        trainer._ngrpo_rollout_cache = None
        trainer._ngrpo_discard_reuse_slots = 0
        trainer._ngrpo_attempted_microstep = 0
        trainer._ngrpo_skip_optimizer_step = False
        trainer._metrics = defaultdict(list)
        model = Model()
        inputs = [{"prompt": "question", "solution": "answer"}]

        with (
            patch.object(Trainer, "_prepare_inputs", lambda self, value: value),
            patch.object(
                trainer_module,
                "unwrap_model_for_generation",
                lambda model, accelerator: nullcontext(model),
            ),
            patch.object(
                trainer_module,
                "maybe_apply_chat_template",
                lambda example, processing_class: {"prompt": example["prompt"]},
            ),
        ):
            first_loss = trainer.compute_loss(model, inputs)
            self.assertTrue(first_loss.requires_grad)
            self.assertEqual(model.generate_calls, 1)
            self.assertIsNotNone(trainer._ngrpo_rollout_cache)
            cached_old = trainer._ngrpo_rollout_cache.old_per_token_logps.clone()

            model.theta.data.fill_(1.0)
            trainer._ngrpo_attempted_microstep = 1
            second_loss = trainer.compute_loss(model, inputs)

        self.assertTrue(second_loss.requires_grad)
        self.assertEqual(model.generate_calls, 1)
        self.assertIsNone(trainer._ngrpo_rollout_cache)
        self.assertTrue(torch.equal(cached_old, cached_old.detach()))
        self.assertEqual(len(trainer._metrics["ngrpo/ratio_abs_deviation"]), 1)
        self.assertGreater(trainer._metrics["ngrpo/ratio_abs_deviation"][-1], 0.0)

    def test_trainer_sampler_wiring_repeats_each_rank_batch(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer._train_batch_size = 1
        trainer.accelerator = SimpleNamespace(num_processes=2)
        trainer.args = SimpleNamespace(
            gradient_accumulation_steps=1,
            data_seed=7,
            seed=42,
        )
        trainer.train_dataset = range(8)
        trainer.ngrpo_num_iterations = 2

        def shard(rank):
            sampler = trainer._get_train_sampler()
            batches = BatchSampler(sampler, batch_size=1, drop_last=False)
            sharded = BatchSamplerShard(
                batches,
                num_processes=2,
                process_index=rank,
                split_batches=False,
            )
            return [batch[0] for batch in sharded]

        rank_zero = shard(0)
        rank_one = shard(1)
        self.assertTrue(all(a == b for a, b in zip(rank_zero[::2], rank_zero[1::2])))
        self.assertTrue(all(a == b for a, b in zip(rank_one[::2], rank_one[1::2])))
        self.assertTrue(all(a != b for a, b in zip(rank_zero[::2], rank_one[::2])))

    def test_training_step_intercepts_only_skipped_backward(self):
        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.loss_type = "ngrpo"
        trainer._ngrpo_step_gate = grpo_variants.NGRPOStepGate()
        trainer._ngrpo_skip_optimizer_step = False
        trainer._ngrpo_attempted_microstep = 0
        trainer.state = SimpleNamespace(global_step=3)
        calls = []
        trainer.accelerator = SimpleNamespace(
            backward=lambda loss, **kwargs: calls.append(loss.item())
        )

        def parent_training_step(self, model, inputs, num_items_in_batch=None):
            self._ngrpo_skip_optimizer_step = inputs["skip"]
            self.accelerator.backward(torch.tensor(1.0, requires_grad=True))
            return torch.tensor(1.0)

        with patch.object(Trainer, "training_step", parent_training_step):
            trainer.training_step(None, {"skip": False})
            self.assertEqual(calls, [1.0])
            self.assertEqual(trainer._ngrpo_attempted_microstep, 1)

            trainer.training_step(None, {"skip": True})
            self.assertEqual(calls, [1.0])
            self.assertTrue(trainer._ngrpo_step_gate.armed)
            self.assertEqual(trainer._ngrpo_attempted_microstep, 2)

        callback = _NGRPOSkipUpdateCallback(trainer)
        control = SimpleNamespace(
            should_log=True,
            should_save=True,
            should_evaluate=True,
        )
        callback.on_optimizer_step(None, trainer.state, control)
        trainer.state.global_step += 1
        callback.on_step_end(None, trainer.state, control)
        self.assertEqual(trainer.state.global_step, 3)
        self.assertFalse(trainer._ngrpo_skip_optimizer_step)
        self.assertFalse(control.should_log)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)

    def test_reused_loss_uses_fixed_old_logps_and_clears_cache(self):
        class Accelerator:
            num_processes = 1

            @staticmethod
            def reduce(value, reduction="sum"):
                return value

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, **kwargs):
                batch, length = input_ids.shape
                logits = torch.zeros(batch, length, 3)
                logits[..., 0] = self.theta
                return SimpleNamespace(logits=logits)

        trainer = object.__new__(Qwen2VLGRPOTrainer)
        trainer.accelerator = Accelerator()
        trainer.num_generations = 2
        trainer.epsilon_low = 0.16
        trainer.epsilon_high = 0.24
        trainer._metrics = defaultdict(list)
        trainer._ngrpo_skip_optimizer_step = False

        model = Model()
        prompt_completion_ids = torch.zeros((2, 3), dtype=torch.long)
        old_logps, _ = trainer._get_per_token_logps(
            model, prompt_completion_ids, return_full_logps=False
        )
        cached_old_logps = old_logps.detach().clone()
        model.theta.data.fill_(1.0)
        trainer._ngrpo_rollout_cache = _NGRPORolloutCache(
            prompt_completion_ids=prompt_completion_ids,
            prompt_inputs={},
            prompt_length=1,
            completion_mask=torch.ones_like(old_logps, dtype=torch.int),
            old_per_token_logps=cached_old_logps,
            advantages=torch.tensor([1.0, -1.0]),
            all_correct_groups=torch.tensor([False]),
            rollout_metrics={},
            remaining_reuses=1,
            batch_fingerprint=(),
        )

        loss = trainer._compute_reused_ngrpo_loss(model, phase=1)
        self.assertTrue(loss.requires_grad)
        self.assertIsNone(trainer._ngrpo_rollout_cache)
        self.assertGreater(trainer._metrics["ngrpo/ratio_abs_deviation"][-1], 0.0)
        self.assertGreater(trainer._metrics["ngrpo/clip_fraction"][-1], 0.0)
        self.assertIsNone(cached_old_logps.grad_fn)


if __name__ == "__main__":
    unittest.main()
