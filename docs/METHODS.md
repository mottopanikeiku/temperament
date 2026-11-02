# Methods

Python 3.13 and `uv`. The lockfile pins everything. Reports run on a CPU from the saved per-item answers; re-scoring models needs an H100 80GB on Modal (paid).

## Accuracy study

**Items.** All 1,172 ARC-Challenge test items, 2,000 uniformly sampled HellaSwag validation items, and 2,000 MMLU test items stratified by subject. Seed 20261007; `data/accuracy/selection.json` lists every item ID.

**Prompts and scoring.** Zero-shot, no chat template, following lm-evaluation-harness v0.4.9.2. ARC and HellaSwag use length-normalized continuation log-likelihood (`acc_norm`); MMLU compares A–D label likelihoods (`acc`). Batches hold equal-length inputs without padding.

**Attention.** Every layer's attention goes through FA3's E4M3 API, with one kernel call per layer. Projections, RoPE, norms and MLPs stay BF16, and the baseline is native BF16 SDPA. Variants, named as in the result files:

| Name | Q/K preprocessing |
|---|---|
| `tile` | none, plain FA3 FP8 with full-sequence per-head scales |
| `rotate` | shared signed Hadamard on Q and K (V is never rotated) |
| `smooth_k` | subtract the per-head mean key |
| `rotate_smooth_k` | subtract the mean key, then rotate |

Scales and means use the whole sequence, including later tokens, so scoring is per batch and not streaming.

**Statistics.** Each variant is compared with BF16 on identical items using 20,000 paired bootstrap resamples and percentile 95% intervals. A checkpoint/task is flagged when the drop is at least 2 points and the interval's upper end is below zero. The four direct comparisons (rotate vs plain, center vs plain, center vs rotated, rotate vs centered) pair the same items and change one thing at a time. There is no multiple-comparison correction.

**Models.** `data/accuracy/models.json` pins the first six checkpoints; Qwen2.5-14B and SmolLM2-1.7B come from `qwen14-plan.json` and `smol17-plan.json`. Qwen2.5-3B is under the noncommercial Qwen research license; the rest are Apache-2.0.

**Rebuild the report (CPU):**

```sh
uv sync --locked --python 3.13
uv run python -m study.accuracy.report \
  --additional-plan data/accuracy/qwen14-plan.json \
  --additional-plan data/accuracy/smol17-plan.json
```

This rewrites `results/accuracy/summary.json` and the figures, and prints every paired comparison.

**Re-score on an H100** (allocates paid cloud resources; `--plan` picks the model set):

```sh
uv sync --locked --extra capture --extra hardware --python 3.13
ATTENTION_GPU=none ATTENTION_MINUTES=20 uv run modal run study/accuracy/modal_app.py --mode weights --output data/accuracy
ATTENTION_GPU=none ATTENTION_MINUTES=10 uv run modal run study/accuracy/modal_app.py --mode prepare --output data/accuracy
ATTENTION_GPU=none ATTENTION_MINUTES=10 uv run modal run study/accuracy/modal_app.py --mode tokens --output data/accuracy
ATTENTION_GPU=H100 ATTENTION_MINUTES=55 uv run modal run study/accuracy/modal_app.py --mode full --token-budget 65536 --output results/accuracy
```

Finished model/variant outputs are skipped on rerun. `--mode cleanup` deletes downloaded checkpoints and keeps tokens and answers. `results/accuracy/cost.json` has the compute cost.

## Per-head and loss study (FA3 and SageAttention)

`study/hardware/` runs the FA3 FP8 kernel on Hopper and SageAttention2's INT8-QK/FP8-PV kernel on Ada against the CPU emulator, per physical head, then measures next-token loss with attention replaced in every layer of Qwen2.5-0.5B and 1.5B. The FP64 reference expands the original BF16 operands exactly. Both Qwens include every head; other models contribute the 32 highest predicted heads plus 32 uniformly sampled ones. `uv run --extra capture python -m study.hardware.report` rebuilds the summary. Running it on GPUs goes through `study/hardware/modal_app.py`.

## CPU emulator and six-model head study

`attention.py` emulates one attention head: E4M3/E5M2/BF16 storage, per-block scales, FP32 or reduced-precision accumulation and online softmax, checked against an FP64 reference. `sweep.py` and `figures.py` run synthetic Gaussian sweeps; `diagnosis.py` and `real.py` use Qwen2.5-0.5B operands from *Alice in Wonderland* (`data/alice.txt`).

`study/pipeline.py` captures post-RoPE Q/K/V at every layer of six small models on three public-domain texts, then measures rotation's effect per head and predicts it from quantization residuals (`study/prediction.py`, no fitted coefficient). Rotation raised output error in 250 of 2,880 heads; the predictor ranks harmed heads at 0.978 AUC on three held-out models. The Pages explorer plots these heads.

```sh
uv sync --locked --extra capture --python 3.13
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python sweep.py --study smoke --seeds 3 --output /tmp/attention-smoke
uv run --extra capture python -m study.pipeline --model all --seconds 500   # repeat until complete
uv run python -m study.report --results-dir results/v2
```

`data/regression/cases.npz` holds five small causal cases (harmed and helped Qwen heads, constant-bias controls) with FP64 references and exact E4M3 bytes. `uv run python -m study.regression` regenerates them.

## Prior work

- FlashAttention-3 (Shah et al., 2024), §3.3: FP8 with Hadamard incoherent processing of Q and K. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- SageAttention (Zhang et al., ICLR 2025), §4.2: smooth-K, subtracting the mean key before quantization. [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)
- SageAttention2 (Zhang et al., ICML 2025): per-block query smoothing with a correction term. [arXiv:2411.10958](https://arxiv.org/abs/2411.10958)
- QuaRot [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) and SpinQuant [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) use rotations to quantize weights, activations and KV caches.
