<div align="center">

# SOVA-TW-GRPO

**Reinforcing Video Reasoning with Focused Thinking via Three Granularities:<br/>Tokens, Answer Sets, and Response Groups**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-bf16-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-Qwen2.5--VL--7B-6E56CF.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![arXiv](https://img.shields.io/badge/arXiv-2505.24718-B31B1B.svg)](https://arxiv.org/abs/2505.24718)

</div>

---

Source release for **SOVA-TW-GRPO**, a group-relative reinforcement learning framework for video
reasoning that allocates credit at **three complementary granularities**: individual **tokens**,
**answer sets**, and whole **response groups**.

It extends **TW-GRPO** (ECCV 2026, *Reinforcing Video Reasoning with Focused Thinking*) with
**SOVA** (*Strict Outcome-Conditioned Virtual Advantages*), which restores a directional learning
signal in response groups whose outcomes are homogeneous, a failure mode inherited from
group-relative normalization.

## Contents

- [Method at a glance](#method-at-a-glance)
- [What SOVA adds](#what-sova-adds)
- [Paper-to-code map](#paper-to-code-map)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Question-Answer Inversion (QAI)](#question-answer-inversion-qai)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Method at a glance

<p align="center">
  <img src="docs/figs/motivation.png" alt="Motivation for SOVA-TW-GRPO" width="100%">
</p>

Group Relative Policy Optimization scores a whole response by the correctness of its final answer
and normalizes that score within a sampled group. One scalar is therefore shared by every token of
a reasoning chain, by every answer that is not exactly correct, and by every response of a group
whose outcomes agree. SOVA-TW-GRPO refines credit assignment at each of those three levels.

| Granularity | Mechanism | Effect | Introduced in |
| :-- | :-- | :-- | :-- |
| **Token** | Importance weighting from intra-group information entropy | Prioritizes tokens with high information density over generic scaffolding | TW-GRPO (ECCV 2026) |
| **Answer set** | Multi-level soft reward over multi-answer QA, with Question-Answer Inversion for data augmentation | Distinguishes partial correctness instead of a binary 0/1 signal | TW-GRPO (ECCV 2026) |
| **Response group** | **SOVA**: strict outcome-conditioned virtual advantages | Restores a directional signal where group-relative normalization collapses | **This work** |

## What SOVA adds

When every response in a sampled group is fully correct and well formatted, or when none earns any
accuracy credit, the group-relative advantage is zero for every response and no gradient survives,
precisely at the outcome extremes that carry the clearest correctness evidence. Token weighting
and soft rewards cannot recover it, because both act on an advantage that has already vanished.

SOVA inserts a **signed virtual anchor** into the group-normalization statistics on those two
strict conditions only. The anchor contributes no response tokens; it shifts only the mean and
standard deviation used to normalize the real responses. The corrected advantage is a **residual
interpolation** between the frozen TW-GRPO advantage and the virtual-statistics advantage,
controlled by a per-branch coefficient `lambda`.

| Branch | Gate condition | Anchor, relative to the group's common reward | Induced signal |
| :-- | :-- | :-- | :-- |
| **Positive** | Every response fully correct **and** well formatted | `1.5`, below the common reward of `2.0` | Positive advantage |
| **Negative** | No response earns accuracy credit | `2.0`, above the common reward (`f_i <= 1`) | Negative advantage |

<p align="center">
  <img src="docs/figs/sova_qualitative.png" alt="SOVA advantage recalibration on an all-zero-accuracy group" width="100%">
</p>

The anchors set only a **direction**, not a magnitude: the induced group-level shift is known in
closed form, is bounded independently of the anchor value, and preserves the reward ordering among
sampled responses. Treat `SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD` and
`SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD` as semantic constants rather than strength knobs; use
`lambda` to control the strength.

## Paper-to-code map

| Paper | Component | Code |
| :-- | :-- | :-- |
| Sec. III-B | Token importance weight `w_t` from intra-group KL divergence | [`src/open_r1/trainer/grpo_trainer.py`](src/open_r1/trainer/grpo_trainer.py) → `compute_token_importance_kl_logs_uniform()`; applied to the per-token loss. Flag: `--alpha` |
| Sec. III-C, Eq. (7) | Multi-level soft accuracy reward | [`src/open_r1/grpo.py`](src/open_r1/grpo.py) → `accuracy_reward()` (soft) vs. `origin_accuracy_reward()` (binary), `format_reward()`. Flags: `--reward_funcs`, `--question_type` |
| Sec. III-C | Question-Answer Inversion (QAI) | [`data/question_answer_inverse/`](data/question_answer_inverse/) → `convert_nextgqa.py`, `convert_star.py` |
| Sec. III-D, Eq. (10) | Strict outcome conditions (the two gates) | [`src/open_r1/sova.py`](src/open_r1/sova.py) → `apply_sova_advantages()` |
| Sec. III-D, Eq. (17) | Residual interpolation between baseline and virtual-statistics advantage | [`src/open_r1/sova.py`](src/open_r1/sova.py) → `apply_sova_advantages()`, `_virtual_group_statistics()` |
| Sec. IV | Configuration validity conditions used by the analysis | [`src/open_r1/sova.py`](src/open_r1/sova.py) → `validate_sova_configuration()` |
| Sec. II-C, Table I | NGRPO and AVSPO baselines for degenerate advantage groups | [`src/open_r1/grpo_variants.py`](src/open_r1/grpo_variants.py) → `compute_ngrpo_advantages()`, AVSPO transforms. Flag: `--loss_type ngrpo` / `avspo` |

## Repository layout

```
SOVA-TW-GRPO/
├── configs/
├── data/
│   └── question_answer_inverse/
├── example/
├── scripts/
│   ├── sova-tw-grpo.sh
│   ├── tw-grpo.sh
│   ├── eval-sova-tw-grpo.sh
│   └── eval-sova-general.sh
├── src/
│   ├── eval/
│   └── open_r1/
│       ├── grpo.py
│       ├── grpo_variants.py
│       ├── sova.py
│       └── trainer/
└── tests/
```

## Setup

> [!NOTE]
> The training commands below are configured for one node with 2 x H800 (80 GB).
> Training for 500 steps takes roughly 4 hours.

### Step 1: Environment

```bash
git clone https://github.com/just-a-go/SOVA-TW-GRPO.git
cd SOVA-TW-GRPO
conda create -n sova-tw-grpo python=3.10
conda activate sova-tw-grpo
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
pip3 install "qwen-vl-utils>=0.0.10" decord
```

### Step 2: Model backbone

```bash
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir Qwen/Qwen2.5-VL-7B-Instruct
```

### Step 3: Training data (CLEVRER)

Training uses the counterfactual split of CLEVRER. Annotations and videos are not bundled;
download them from the official source into your local `data/CLEVRER` directory.

```bash
mkdir -p data/CLEVRER/{train_video,validation_video}

wget -P data/CLEVRER/train_video http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
unzip data/CLEVRER/train_video/video_train.zip -d data/CLEVRER/train_video
rm data/CLEVRER/train_video/video_train.zip

wget -P data/CLEVRER/validation_video http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
unzip data/CLEVRER/validation_video/video_validation.zip -d data/CLEVRER/validation_video
rm data/CLEVRER/validation_video/video_validation.zip
```

### Step 4: Evaluation data (optional)

Skip this step if you only want to reproduce the CLEVRER results.

| Dataset | Size | Link |
| :-- | --: | :-- |
| NExT-QA | 11 GB | [huggingface.co/datasets/lmms-lab/NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA) |
| MMVU | 0.9 GB | [huggingface.co/datasets/yale-nlp/MMVU](https://huggingface.co/datasets/yale-nlp/MMVU) |
| MVBench | 16 GB | [huggingface.co/datasets/OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) |
| TempCompass | 0.4 GB | [huggingface.co/datasets/lmms-lab/TempCompass](https://huggingface.co/datasets/lmms-lab/TempCompass) |
| Video-MME | 94 GB | [huggingface.co/datasets/lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| STAR | 7 GB | [modelscope.cn/datasets/Video-R1/Video-R1-data](https://modelscope.cn/datasets/Video-R1/Video-R1-data/files) |

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/sova-tw-grpo.sh
```

The launcher takes no arguments; configure a run in the paths and selector blocks at the top of
the script. The paths there are from the development machine, so edit them first.

### Method selector

| `LOSS_TYPE` | Method | Reward functions | SOVA |
| :-- | :-- | :-- | :-- |
| `tw_grpo` | TW-GRPO, optionally with SOVA | `accuracy` (soft) + `format` | Available |
| `grpo` | Standard GRPO | `origin_accuracy` (binary) | Off |
| `ngrpo` | NGRPO baseline | `origin_accuracy` (binary) | Off |
| `avspo` | AVSPO baseline | `origin_accuracy` (binary) | Off |

SOVA requires `LOSS_TYPE="tw_grpo"`; the trainer rejects any other combination.

### SOVA settings

| Variable | Values | Meaning |
| :-- | :-- | :-- |
| `SOVA_MODE` | `positive` / `negative` / `bidirectional` | Which gate(s) are active |
| `SOVA_LAMBDA_POSITIVE` | `[0, 1]` | Interpolation weight of the positive branch |
| `SOVA_LAMBDA_NEGATIVE` | `[0, 1]` | Interpolation weight of the negative branch |
| `SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD` | `1.5` (default) | Positive anchor (semantic constant, keep fixed) |
| `SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD` | `2.0` (default) | Negative anchor (semantic constant, keep fixed) |

A `lambda` must be strictly positive exactly when its branch is enabled; `validate_sova_configuration()`
fails closed on any other combination.

### Other training options

| Flag | Values | Meaning |
| :-- | :-- | :-- |
| `--question_type` | `mixed` (default) / `single` | Multi-answer or single-choice QA |
| `--alpha` | `1.70` (default) | Upper bound of the token importance weight |
| `--reward_funcs` | `accuracy` / `origin_accuracy` / `format` | Soft or binary accuracy, plus format reward |
| `--jsonl_path` | path | Training or evaluation JSON |
| `--num_generations` | `8` (default) | Group size `G` |

To reproduce the conference configuration instead, run `bash scripts/tw-grpo.sh`.

## Evaluation

```bash
bash scripts/eval-sova-tw-grpo.sh    # SOVA-TW-GRPO on video reasoning benchmarks
bash scripts/eval-sova-general.sh    # General video understanding benchmarks
bash scripts/evaluate.sh             # Conference-version evaluation entry point
```

TW-GRPO checkpoints are available at [Falconss1/TW-GRPO](https://huggingface.co/Falconss1/TW-GRPO).

<details>
<summary><b>Baseline evaluation</b></summary>

```bash
huggingface-cli download --resume-download Video-R1/Video-R1-7B --local-dir Video-R1/Video-R1-7B
huggingface-cli download --resume-download Video-R1/Qwen2.5-VL-7B-COT-SFT --local-dir Video-R1/Qwen2.5-VL-7B-COT-SFT
huggingface-cli download --resume-download OpenGVLab/VideoChat-R1_7B --local-dir OpenGVLab/VideoChat-R1_7B
```

```bash
bash scripts/evaluate_video_r1.sh        # Video-R1
bash scripts/evaluate_qwen2_5vl_sft.sh   # Qwen2.5-VL-7B-COT-SFT
bash scripts/evaluate_videochat_r1.sh    # VideoChat-R1
bash scripts/evaluate_qwen2_5vl.sh       # Qwen2.5-VL, zero-shot
```

For other baselines, change `MODEL_NAME` inside the evaluation script.

NGRPO and AVSPO are run through the training launcher with `LOSS_TYPE="ngrpo"` or `"avspo"`.

</details>

## Question-Answer Inversion (QAI)

QAI converts single-choice QA into multi-answer QA by negating the question and inverting the
answer set, which supplies the multi-level reward with training data. A worked example is given in
[`example/tutorial/qai_tutorial.md`](example/tutorial/qai_tutorial.md).

```bash
python data/question_answer_inverse/convert_nextgqa.py   # NExT-GQA
python data/question_answer_inverse/convert_star.py      # STAR
```

Outputs are written to your local `data/evaluation/` directory as `nextgqa_val_mixed.json` and
`STAR_mixed.json`.

Case studies comparing reasoning paths against [Video-R1](https://github.com/tulerfeng/Video-R1)
are collected in [`example/performance_comparison.md`](example/performance_comparison.md).

## Acknowledgements

This work builds on the open-source community, in particular
[Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video),
[Video-R1](https://github.com/tulerfeng/Video-R1) and
[VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1).

## Citation

If you find this project useful, please consider citing the conference version:

```bibtex
@inproceedings{dang2026reinforcing,
  title     = {Reinforcing Video Reasoning with Focused Thinking},
  author    = {Dang, Jisheng and Wu, Jingze and Wang, Teng and Lin, Xuanhui and
               Zhu, Nannan and Chen, Hongbo and Zheng, Wei-Shi and Wang, Meng and
               Chua, Tat-Seng},
  booktitle = {European Conference on Computer Vision (ECCV)},
  publisher = {Springer},
  year      = {2026},
  note      = {arXiv:2505.24718}
}
```

ECCV 2026 takes place 10-12 September 2026; page numbers and the DOI will be added to this entry
once the proceedings are published. The journal extension introducing SOVA is under review and
this section will be updated once it has a citable reference.

## License

Released under the [Apache License 2.0](LICENSE).
