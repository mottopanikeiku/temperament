"""Build a small CPU numerical corpus, not a hardware-kernel conformance suite.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m study.regression
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from attention import emulate, metrics, quantize_qk, raw_storage_cast, reference
from diagnosis import CONFIGS, CONTROL_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def build_corpus(destination):
    """Freeze public captured prefixes and matched synthetic constant-bias controls."""
    source = ROOT / "data/qwen-qkv.npz"
    provenance = json.loads(source.with_suffix(".json").read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != provenance["capture_sha256"]:
        raise ValueError("Capture checksum mismatch")
    cases = {}
    with np.load(source, allow_pickle=False) as captured:
        for layer, effect in ((0, "harmed"), (12, "helped")):
            name = f"qwen_layer{layer}_head0"
            qkv = tuple(captured[f"layer{layer}_head0_{x}"][:128].copy() for x in "qkv")
            cases[name] = (
                qkv,
                {
                    "kind": "captured",
                    "layer": layer,
                    "head": 0,
                    "token_slice": [0, 128],
                    "rotation_effect": effect,
                },
            )
    rng = np.random.default_rng(20261029)
    q, k, v = (rng.integers(-32, 33, size=(128, 64)).astype(np.float32) / 16 for _ in range(3))
    biased = k.copy()
    biased[:, 0] += 32  # Dyadic values preserve the residual exactly in float32.
    constant = np.zeros_like(k)
    constant[:, 0] = 32
    for name, keys in (
        ("control_base", k),
        ("control_bias", biased),
        ("control_constant", constant),
    ):
        cases[name] = ((q, keys, v), {"kind": "synthetic", "seed": 20261029})
    arrays, records = {}, []
    for name, (qkv, metadata) in cases.items():
        arrays.update({f"{name}_{x}": value for x, value in zip("qkv", qkv, strict=True)})
        expected = reference(*qkv, causal=True)
        arrays[f"{name}_reference"] = expected
        errors = {}
        for variant in CONTROL_VARIANTS:
            cfg = CONFIGS[variant]
            output = emulate(*qkv, cfg)
            arrays[f"{name}_{variant}"] = output
            errors[variant] = metrics(output, expected)["relative_frobenius"]
            qs, qscale, ks, kscale = quantize_qk(qkv[0], qkv[1], cfg)
            for operand, values, scale in (("q", qs, qscale), ("k", ks, kscale)):
                arrays[f"{name}_{variant}_{operand}_bits"] = raw_storage_cast(values, "e4m3").view(
                    np.uint8
                )
                arrays[f"{name}_{variant}_{operand}_scale"] = scale
        records.append({"name": name, **metadata, "relative_frobenius": errors})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "cases.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "schema_version": 1,
        "description": "CPU emulator regression examples; not FA3/Sage expected outputs",
        "source": {
            "path": "data/qwen-qkv.npz",
            "sha256": source_hash,
            "model": provenance["model"],
            "revision": provenance["revision"],
            "license": provenance["model_license"],
            "text": provenance["text_source"],
        },
        "shape": [128, 64],
        "causal": True,
        "reference": "Unquantized original Q/K/V, float64 causal softmax(QK^T / sqrt(64)) V",
        "variants": {name: asdict(CONFIGS[name]) for name in CONTROL_VARIANTS},
        "arrays": "<case>_{q,k,v,reference,<variant>,<variant>_{q,k}_{bits,scale}}",
        "cases_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "cases": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/regression")
    args = parser.parse_args()
    manifest = build_corpus(args.output)
    for case in manifest["cases"]:
        errors = case["relative_frobenius"]
        print(f"{case['name']}: tile={errors['e4_tile']:.6f}, rotate={errors['e4_rotate']:.6f}")
    print(f"Wrote {len(manifest['cases'])} CPU cases to {args.output}")


if __name__ == "__main__":
    main()
