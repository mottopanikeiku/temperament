"""Public CPU examples: independent references, invariants, and emulator packing parity."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from attention import Config, center_keys, emulate, quantize_qk, raw_storage_cast
from study.regression import build_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/regression"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text())
NAMES = [case["name"] for case in MANIFEST["cases"]]


@pytest.fixture(scope="module")
def corpus():
    with np.load(CORPUS / "cases.npz", allow_pickle=False) as arrays:
        yield arrays


def dense_reference(q, k, v):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores[np.triu_indices(len(q), 1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ v


@pytest.mark.parametrize("name", NAMES)
def test_public_corpus_reference_and_exact_attention_invariants(corpus, name):
    q, k, v = (corpus[f"{name}_{x}"].astype(np.float64) for x in "qkv")
    expected = corpus[f"{name}_reference"]
    for values in (q, k, v, expected):
        assert values.shape == tuple(MANIFEST["shape"])
        assert np.isfinite(values).all()
    np.testing.assert_allclose(dense_reference(q, k, v), expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        dense_reference(q, center_keys(k), v), expected, rtol=1e-10, atol=1e-11
    )
    # Independent dense float64 Sylvester matrix, not the emulator's fp32 butterflies.
    matrix = np.ones((1, 1))
    while len(matrix) < q.shape[1]:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    signs = np.random.default_rng(1729).choice([-1, 1], size=q.shape[1])
    matrix = signs[:, None] * matrix / np.sqrt(len(matrix))
    for keys in (k, center_keys(k)):
        np.testing.assert_allclose(
            dense_reference(q @ matrix, keys @ matrix, v), expected, rtol=1e-10, atol=1e-11
        )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("variant", MANIFEST["variants"])
def test_public_corpus_emulator_outputs_and_exact_fp8_qk_packing(corpus, name, variant):
    qkv = [corpus[f"{name}_{x}"] for x in "qkv"]
    cfg = Config(**MANIFEST["variants"][variant])
    np.testing.assert_allclose(
        emulate(*qkv, cfg), corpus[f"{name}_{variant}"], rtol=1e-5, atol=1e-6
    )
    qs, qscale, ks, kscale = quantize_qk(qkv[0], qkv[1], cfg)
    for operand, values, scale in (("q", qs, qscale), ("k", ks, kscale)):
        np.testing.assert_array_equal(
            raw_storage_cast(values, "e4m3").view(np.uint8),
            corpus[f"{name}_{variant}_{operand}_bits"],
        )
        np.testing.assert_array_equal(scale, corpus[f"{name}_{variant}_{operand}_scale"])


def test_corpus_contains_both_rotation_directions(corpus):
    effects = set()
    for case in MANIFEST["cases"]:
        if case["kind"] != "captured":
            continue
        name = case["name"]
        expected = corpus[f"{name}_reference"]
        errors = {
            variant: np.linalg.norm(corpus[f"{name}_{variant}"] - expected)
            for variant in ("e4_tile", "e4_rotate")
        }
        effect = "harmed" if errors["e4_rotate"] > errors["e4_tile"] else "helped"
        assert effect == case["rotation_effect"]
        effects.add(effect)
    assert effects == {"harmed", "helped"}


def test_constant_bias_controls_have_independent_known_answers(corpus):
    np.testing.assert_array_equal(corpus["control_base_q"], corpus["control_bias_q"])
    np.testing.assert_array_equal(corpus["control_base_v"], corpus["control_bias_v"])
    bias = np.zeros((128, 64))
    bias[:, 0] = 32
    np.testing.assert_array_equal(corpus["control_bias_k"] - corpus["control_base_k"], bias)
    np.testing.assert_allclose(
        corpus["control_base_reference"],
        corpus["control_bias_reference"],
        rtol=1e-12,
        atol=1e-12,
    )
    values = corpus["control_constant_v"].astype(np.float64)
    prefix_mean = values.cumsum(axis=0) / np.arange(1, len(values) + 1)[:, None]
    np.testing.assert_allclose(
        corpus["control_constant_reference"], prefix_mean, rtol=1e-12, atol=1e-12
    )


def test_corpus_hash_provenance_and_captured_prefixes(corpus):
    assert (
        hashlib.sha256((CORPUS / "cases.npz").read_bytes()).hexdigest() == MANIFEST["cases_sha256"]
    )
    source = ROOT / MANIFEST["source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == MANIFEST["source"]["sha256"]
    with np.load(source, allow_pickle=False) as arrays:
        for case in MANIFEST["cases"]:
            if case["kind"] == "captured":
                for operand in "qkv":
                    np.testing.assert_array_equal(
                        corpus[f"{case['name']}_{operand}"],
                        arrays[f"layer{case['layer']}_head{case['head']}_{operand}"][:128],
                    )


def test_corpus_builder_is_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert build_corpus(first) == build_corpus(second)
    for filename in ("manifest.json", "cases.npz"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
