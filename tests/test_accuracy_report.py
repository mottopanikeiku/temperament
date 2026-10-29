"""Synthetic CPU tests only; compressed measured runs are the primary evidence."""

import copy
import gzip
import hashlib
import itertools
import json
from xml.etree import ElementTree

import numpy as np
import pytest

from study.accuracy.report import (
    INTERVENTIONS,
    VARIANTS,
    build_report,
    intervention_table,
    paired_change,
    paired_figure,
    stream_seed,
    summarize,
    validate_manifest,
    validate_plan,
    validate_run,
)

PLAN_HASH, ITEMS_HASH = "a" * 64, "b" * 64


@pytest.fixture
def frozen():
    plan = {
        "models": [
            {"key": key, "revision": str(index) * 40}
            for index, key in enumerate(("qwen05", "qwen15", "qwen3", "qwen7", "mistral7", "olmo7"))
        ],
        "variants": list(VARIANTS),
        "tasks": {
            "arc_challenge": {"count": 2, "metric": "acc_norm"},
            "hellaswag": {"count": 3, "metric": "acc_norm"},
            "mmlu": {"count": 4, "metric": "acc"},
        },
        "bootstrap": {"replicates": 20000, "seed": 20261007, "interval": 0.95},
        "harm_rule": {
            "minimum_drop": 0.02,
            "interval_upper_below": 0,
            "definition": "drop and negative interval",
        },
    }
    manifest = {"schema_version": 1, "items": []}
    for task, spec in plan["tasks"].items():
        for index in range(spec["count"]):
            choices = ["a", "bbbb"] if task != "mmlu" else ["A", "B"]
            manifest["items"].append(
                {
                    "task": task,
                    "item_id": f"{task}:{index}",
                    "context": "Question: Answer:",
                    "choices": choices,
                    "gold": index % 2,
                    "metric": spec["metric"],
                    "normalization_lengths": [len(choice) for choice in choices],
                }
            )
    runs = {}
    for model in plan["models"]:
        rows = []
        for item in manifest["items"]:
            lengths = item["normalization_lengths"]
            likelihoods = [-2.0, -4.0]
            scores = [ll / length for ll, length in zip(likelihoods, lengths, strict=True)]
            if item["metric"] == "acc":
                scores = likelihoods.copy()
            prediction = int(np.argmax(scores))
            for variant in VARIANTS:
                rows.append(
                    {
                        "model": model["key"],
                        "task": item["task"],
                        "item_id": item["item_id"],
                        "variant": variant,
                        "gold": item["gold"],
                        "prediction": prediction,
                        "correct": prediction == item["gold"],
                        "choice_log_likelihoods": likelihoods.copy(),
                        "choice_scores": scores.copy(),
                        "normalization_lengths": lengths.copy(),
                        "choice_token_counts": [1, 2],
                        "context_tokens": 5,
                        "sequence_lengths": [5, 6],
                    }
                )
        runs[model["key"]] = {
            "schema_version": 1,
            "model": model["key"],
            "revision": model["revision"],
            "plan_sha256": PLAN_HASH,
            "item_manifest_sha256": ITEMS_HASH,
            "variants": list(VARIANTS),
            "runtime": {"hardware": "synthetic-test-only"},
            "adapter": {"name": "synthetic-test-only"},
            "rows": rows,
        }
    return plan, manifest, runs


def checked(frozen):
    plan, manifest, runs = frozen
    validate_plan(plan)
    items = validate_manifest(plan, manifest)
    indexed = {
        model["key"]: validate_run(runs[model["key"]], model, plan, items, PLAN_HASH, ITEMS_HASH)
        for model in plan["models"]
    }
    return plan, items, indexed


def test_identical_correlated_pairs_have_exact_zero_interval():
    # Marginal accuracies each have sampling variance; their paired difference does not.
    baseline = np.array([False, True] * 50)
    result = paired_change(baseline, baseline.copy(), seed=19)
    assert result["baseline_accuracy"] == result["accuracy"] == 0.5
    assert result["delta"] == 0
    assert result["ci95"] == [0, 0]
    assert result["beneficial"] == result["harmed"] == 0
    assert result["tied"] == result["denominator"] == 100


@pytest.mark.parametrize("candidate, expected", [(False, -1), (True, 1)])
def test_constant_paired_changes_have_degenerate_interval(candidate, expected):
    result = paired_change([not candidate] * 6, [candidate] * 6, seed=22)
    assert result["delta"] == expected
    assert result["ci95"] == [expected, expected]


def test_varying_pairs_match_analytic_item_resampling_and_are_deterministic():
    baseline, candidate = [True, True, False, False], [False, True, False, True]
    seed = stream_seed(20261007, "qwen7", "mmlu", "tile")
    first = paired_change(baseline, candidate, seed)
    assert first == paired_change(baseline, candidate, seed)
    # Enumerate the exact 4^4 equally likely ordered item bootstrap samples.
    differences = [-1, 0, 0, 1]
    exact = sorted(
        sum(differences[index] for index in indices) / 4
        for indices in itertools.product(range(4), repeat=4)
    )
    assert first["ci95"] == list(np.quantile(exact, [0.025, 0.975])) == [-0.75, 0.75]
    assert (first["harmed"], first["tied"], first["beneficial"]) == (1, 2, 1)
    assert first["delta"] == 0
    assert first["accuracy"] == first["baseline_accuracy"] == 0.5
    assert seed != stream_seed(20261007, "qwen7", "mmlu", "rotate")
    assert seed != stream_seed(20261007, "qwen3", "mmlu", "tile")
    assert seed != stream_seed(20261007, "qwen7", "hellaswag", "tile")
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    "baseline,candidate", [([], []), ([True], []), ([1], [0]), ([float("nan")], [True])]
)
def test_invalid_paired_inputs_rejected(baseline, candidate):
    with pytest.raises(ValueError):
        paired_change(baseline, candidate, 42)


def test_all_models_tasks_variants_metrics_and_denominators_reported(frozen):
    plan, items, runs = checked(frozen)
    summary = summarize(plan, items, runs)
    assert summary["complete"] is True
    assert len(summary["rows"]) == 6 * 3 * 5
    assert summary["primary_metrics"] == {
        "arc_challenge": "acc_norm",
        "hellaswag": "acc_norm",
        "mmlu": "acc",
    }
    for row in summary["rows"]:
        assert row["denominator"] == plan["tasks"][row["task"]]["count"]
        assert row["delta"] == 0 and row["ci95"] == [0, 0]
        assert not row["harm_flag"]
    assert summary["sample_limitations"]
    json.dumps(summary, allow_nan=False)
    figure = paired_figure(summary)
    assert len(figure.encode()) < 400000
    root = ElementTree.fromstring(figure)
    assert not root.findall(".//{http://www.w3.org/2000/svg}image")
    assert len(root.findall(".//{http://www.w3.org/2000/svg}g[@id]")) >= 6 * 3 * 4
    for model in plan["models"]:
        for task in plan["tasks"]:
            for variant in VARIANTS[1:]:
                assert f"{model['key']} / {task} / {variant}" in figure
                assert (
                    root.find(
                        f".//{{http://www.w3.org/2000/svg}}g"
                        f"[@id='cell-{model['key']}-{task}-{variant}']"
                    )
                    is not None
                )
    assert "minus two percentage points" in figure


def test_direct_contrasts_cover_every_model_task_and_four_interventions(frozen):
    plan, items, runs = checked(frozen)
    summary = summarize(plan, items, runs)
    assert len(summary["interventions"]) == 6 * 3 * 4
    assert {
        (row["model"], row["task"], row["candidate_variant"], row["baseline_variant"])
        for row in summary["interventions"]
    } == {
        (model["key"], task, candidate, baseline)
        for model in plan["models"]
        for task in plan["tasks"]
        for candidate, baseline in INTERVENTIONS
    }
    for row in summary["interventions"]:
        assert row["denominator"] == plan["tasks"][row["task"]]["count"]
        assert row["delta"] == 0 and row["ci95"] == [0, 0]
        assert row["primary_metric"] == plan["tasks"][row["task"]]["metric"]
        assert row["revision"] == next(
            model["revision"] for model in plan["models"] if model["key"] == row["model"]
        )
        assert "harm_flag" not in row  # BF16-relative flags do not attribute rotation.


def test_direct_contrasts_pair_by_task_and_id_not_record_order(frozen):
    plan, manifest, raw = frozen
    # Reuse IDs across tasks; the pairing key must include both fields.
    for item in manifest["items"]:
        item["item_id"] = item["item_id"].split(":")[-1]
    answers = {
        "bf16": [True, True, True, True],
        "tile": [True, False, True, False],
        "rotate": [False, False, True, False],
        "smooth_k": [True, True, True, False],
        "rotate_smooth_k": [True, True, False, True],
    }
    for run in raw.values():
        for row in run["rows"]:
            row["item_id"] = row["item_id"].split(":")[-1]
            if row["task"] != "mmlu":
                continue
            correct = answers[row["variant"]][int(row["item_id"])]
            prediction = row["gold"] if correct else 1 - row["gold"]
            scores = [-2.0, -2.0]
            scores[prediction] = -1.0
            row.update(
                correct=correct,
                prediction=prediction,
                choice_log_likelihoods=scores.copy(),
                choice_scores=scores.copy(),
            )
        np.random.default_rng(87).shuffle(run["rows"])
    checked_plan, items, runs = checked(frozen)
    summary = summarize(checked_plan, items, runs)
    rows = [
        row for row in summary["interventions"] if row["model"] == "qwen7" and row["task"] == "mmlu"
    ]
    expected_counts = [(0, 1, 3), (1, 0, 3), (3, 1, 0), (1, 1, 2)]
    for row, (candidate, baseline), counts in zip(
        rows, INTERVENTIONS, expected_counts, strict=True
    ):
        seed = stream_seed(
            plan["bootstrap"]["seed"], "qwen7", "mmlu", ["intervention", candidate, baseline]
        )
        expected = paired_change(answers[baseline], answers[candidate], seed)
        assert {key: row[key] for key in expected} == expected
        assert (row["beneficial"], row["harmed"], row["tied"]) == counts
        assert row["bootstrap_seed"] == seed
    assert [row["delta"] for row in rows] == [-0.25, 0.25, 0.5, 0]
    # Same marginal accuracy is not a zero-variance paired effect.
    assert rows[-1]["ci95"][0] < 0 < rows[-1]["ci95"][1]
    # Reordering models, tasks, item IDs and raw records changes no row or stream.
    plan["models"].reverse()
    plan["tasks"] = dict(reversed(list(plan["tasks"].items())))
    manifest["items"].reverse()
    for run in raw.values():
        run["rows"].reverse()
    reordered = summarize(*checked(frozen))

    def identity(row):
        return row["model"], row["task"], row["candidate_variant"], row["baseline_variant"]

    assert sorted(summary["interventions"], key=identity) == sorted(
        reordered["interventions"], key=identity
    )


def test_intervention_table_formats_percentage_points_and_both_interval_endpoints():
    rows = [
        {
            "model": "qwen7",
            "checkpoint": "Qwen2.5-7B-Instruct",
            "task": "hellaswag",
            "candidate_variant": candidate,
            "baseline_variant": baseline,
            "delta": -0.182,
            "ci95": [-0.2095, -0.1545],
        }
        for candidate, baseline in INTERVENTIONS
    ]
    table = intervention_table({"interventions": rows})
    assert table.splitlines()[0] == (
        "| Checkpoint / task | Rotate − plain | Center − plain | "
        "Center rotated − rotated | Rotate centered − centered |"
    )
    assert table.splitlines()[2] == (
        "| Qwen2.5-7B-Instruct / HellaSwag | " + " | ".join(["−18.20 [−20.95, −15.45]"] * 4) + " |"
    )


def test_harm_flag_requires_both_conditions_and_includes_exact_threshold(frozen, monkeypatch):
    plan, items, runs = checked(frozen)
    conditions = itertools.cycle([(-0.02, -0.001), (-0.019, -0.001), (-0.03, 0), (-0.1, 0.01)])

    def change(*args):
        delta, upper = next(conditions)
        return {"delta": delta, "ci95": [delta - 0.1, upper]}

    monkeypatch.setattr("study.accuracy.report.paired_change", change)
    summary = summarize(plan, items, runs)
    assert [row["harm_flag"] for row in summary["rows"][:4]] == [True, False, False, False]


def test_manifest_incomplete_and_duplicate_coverage_rejected(frozen):
    plan, manifest, _ = frozen
    missing = copy.deepcopy(manifest)
    missing["items"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_manifest(plan, missing)
    duplicate = copy.deepcopy(manifest)
    duplicate["items"].append(duplicate["items"][0])
    with pytest.raises(ValueError, match="Duplicate manifest"):
        validate_manifest(plan, duplicate)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("gold", 1, "gold mismatch"),
        ("correct", True, "Correctness"),
        ("choice_log_likelihoods", [-2, float("nan")], "finite"),
        ("choice_scores", [-2, float("inf")], "finite"),
        ("choice_log_likelihoods", [-2], "finite"),
        ("choice_scores", [-2, -4], "normalization"),
        ("normalization_lengths", [1, 5], "normalization lengths"),
        ("choice_token_counts", [True, 2], "choice_token_counts"),
        ("context_tokens", 0, "context_tokens"),
        ("sequence_lengths", [5, 7], "Sequence lengths"),
        ("prediction", True, "prediction"),
        ("variant", "unknown", "variant"),
        ("item_id", "unknown", "Unknown raw item"),
    ],
)
def test_malformed_scores_and_records_rejected(frozen, field, value, message):
    plan, manifest, runs = frozen
    run = runs[plan["models"][0]["key"]]
    run["rows"][0][field] = value
    with pytest.raises(ValueError, match=message):
        validate_run(
            run, plan["models"][0], plan, validate_manifest(plan, manifest), PLAN_HASH, ITEMS_HASH
        )


@pytest.mark.parametrize(
    "mutation,message", [("missing", "coverage"), ("duplicate", "Duplicate raw")]
)
def test_raw_row_coverage_and_duplicates_rejected(frozen, mutation, message):
    plan, manifest, runs = frozen
    run = runs[plan["models"][0]["key"]]
    if mutation == "missing":
        run["rows"].pop()
    else:
        run["rows"].append(copy.deepcopy(run["rows"][0]))
    with pytest.raises(ValueError, match=message):
        validate_run(
            run, plan["models"][0], plan, validate_manifest(plan, manifest), PLAN_HASH, ITEMS_HASH
        )


@pytest.mark.parametrize("field", ["revision", "plan_sha256", "item_manifest_sha256"])
def test_wrong_revision_or_artifact_hash_rejected(frozen, field):
    plan, manifest, runs = frozen
    run = runs[plan["models"][0]["key"]]
    run[field] = "wrong"
    with pytest.raises(ValueError, match="revision|hash"):
        validate_run(
            run, plan["models"][0], plan, validate_manifest(plan, manifest), PLAN_HASH, ITEMS_HASH
        )


def test_normalized_argmax_differs_from_raw_argmax_and_first_tie_wins(frozen):
    plan, manifest, runs = frozen
    model = plan["models"][0]
    run = runs[model["key"]]
    row = run["rows"][0]
    assert row["prediction"] == 1  # normalized -2 < -1, but raw -2 > -4
    row.update(choice_log_likelihoods=[-1, -4], choice_scores=[-1, -1], prediction=0, correct=True)
    items = validate_manifest(plan, manifest)
    validate_run(run, model, plan, items, PLAN_HASH, ITEMS_HASH)
    row.update(prediction=1, correct=False)
    with pytest.raises(ValueError, match="first score argmax"):
        validate_run(run, model, plan, items, PLAN_HASH, ITEMS_HASH)


def test_mmlu_uses_unnormalized_label_likelihood(frozen):
    plan, manifest, runs = frozen
    model = plan["models"][0]
    run = runs[model["key"]]
    row = next(row for row in run["rows"] if row["task"] == "mmlu")
    row["choice_scores"] = [-1, -2]
    with pytest.raises(ValueError, match="normalization"):
        validate_run(run, model, plan, validate_manifest(plan, manifest), PLAN_HASH, ITEMS_HASH)


def test_tokenization_must_match_bf16_for_each_paired_item(frozen):
    plan, manifest, runs = frozen
    model = plan["models"][0]
    run = runs[model["key"]]
    row = run["rows"][1]
    row.update(choice_token_counts=[2, 2], sequence_lengths=[6, 6])
    with pytest.raises(ValueError, match="Variant tokenization"):
        validate_run(run, model, plan, validate_manifest(plan, manifest), PLAN_HASH, ITEMS_HASH)


def test_model_coverage_rejected_by_summary(frozen):
    plan, items, runs = checked(frozen)
    runs.pop("olmo7")
    with pytest.raises(ValueError, match="model coverage"):
        summarize(plan, items, runs)


def write_artifacts(tmp_path, frozen):
    plan, manifest, runs = frozen
    plan_path, items_path = tmp_path / "plan.json", tmp_path / "items.json.gz"
    results = tmp_path / "results"
    results.mkdir()
    plan_path.write_text(json.dumps(plan))
    items_path.write_bytes(gzip.compress(json.dumps(manifest).encode(), mtime=0))
    for run in runs.values():
        run["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        run["item_manifest_sha256"] = hashlib.sha256(items_path.read_bytes()).hexdigest()
        (results / f"{run['model']}.json.gz").write_bytes(
            gzip.compress(json.dumps(run).encode(), mtime=0)
        )
    return results, plan_path, items_path


def test_build_report_preserves_raw_evidence_and_publishes_complete_outputs(tmp_path, frozen):
    results, plan, items = write_artifacts(tmp_path, frozen)
    before = {path.name: path.read_bytes() for path in results.glob("*.json.gz")}
    summary = build_report(results, plan, items)
    assert json.loads((results / "summary.json").read_text()) == summary
    assert (results / "paired-change.svg").stat().st_size < 400000
    assert len(summary["primary_evidence"]) == 6
    assert summary["item_manifest_sha256"] == hashlib.sha256(items.read_bytes()).hexdigest()
    for evidence in summary["primary_evidence"]:
        assert evidence["sha256"] == hashlib.sha256(before[evidence["path"]]).hexdigest()
        assert (results / evidence["path"]).read_bytes() == before[evidence["path"]]


def test_known_pilot_namespace_coexists_but_unknown_full_files_are_rejected(tmp_path, frozen):
    results, plan, items = write_artifacts(tmp_path, frozen)
    (results / "pilot-qwen7.json.gz").write_bytes(
        gzip.compress(json.dumps({"mode": "pilot", "rows": []}).encode(), mtime=0)
    )
    summary = build_report(results, plan, items)
    assert len(summary["primary_evidence"]) == 6
    (results / "unexpected.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(ValueError, match="extra"):
        build_report(results, plan, items)


@pytest.mark.parametrize("mismatch", [None, "parent", "tasks", "duplicate"])
def test_additional_plan_preserves_original_plan_and_rejects_design_changes(
    tmp_path, frozen, mismatch
):
    results, plan_path, items = write_artifacts(tmp_path, frozen)
    plan, _, runs = frozen
    before = {path.name: path.read_bytes() for path in results.glob("*.json.gz")}
    extension = copy.deepcopy(plan)
    extra_key = "qwen05" if mismatch == "duplicate" else "qwen14"
    extension["models"] = [{**plan["models"][0], "key": extra_key}]
    extension.update(
        planned_model_count=1,
        extends_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        results_subdirectory="qwen14",
    )
    if mismatch == "parent":
        extension["extends_plan_sha256"] = "f" * 64
    elif mismatch == "tasks":
        extension["tasks"]["arc_challenge"]["count"] += 1
    extension_path = tmp_path / "qwen14-plan.json"
    extension_path.write_text(json.dumps(extension))
    extra_run = copy.deepcopy(runs["qwen05"])
    extra_run["model"] = extra_key
    extra_run["plan_sha256"] = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    for row in extra_run["rows"]:
        row["model"] = extra_key
    (results / "qwen14").mkdir()
    (results / "qwen14" / f"{extra_key}.json.gz").write_bytes(
        gzip.compress(json.dumps(extra_run).encode(), mtime=0)
    )
    if mismatch:
        with pytest.raises(ValueError):
            build_report(results, plan_path, items, [extension_path])
        assert not (results / "summary.json").exists()
        return
    summary = build_report(results, plan_path, items, [extension_path])
    assert len(summary["rows"]) == 105
    assert len(summary["interventions"]) == 7 * 3 * 4
    assert len([row for row in summary["interventions"] if row["model"] == extra_key]) == 12
    assert len(summary["primary_evidence"]) == 7
    assert summary["primary_evidence"][-1]["path"] == "qwen14/qwen14.json.gz"
    assert summary["additional_plans"][0]["sha256"] == extra_run["plan_sha256"]
    for name, raw in before.items():
        assert (results / name).read_bytes() == raw


def test_missing_model_file_never_publishes_partial_summary(tmp_path, frozen):
    results, plan, items = write_artifacts(tmp_path, frozen)
    (results / "olmo7.json.gz").unlink()
    with pytest.raises(ValueError, match="model file coverage"):
        build_report(results, plan, items)
    assert not (results / "summary.json").exists()
    assert not (results / "paired-change.svg").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("models", []),
        ("variants", list(VARIANTS[:-1])),
        ("bootstrap", {"replicates": 19999, "seed": 20261007}),
        ("harm_rule", {"minimum_drop": 0.01, "interval_upper_below": 0}),
    ],
)
def test_changed_plan_design_rejected(frozen, field, value):
    plan, _, _ = frozen
    plan[field] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_swapped_primary_metrics_rejected(frozen):
    plan, _, _ = frozen
    plan["tasks"]["arc_challenge"]["metric"] = "acc"
    plan["tasks"]["mmlu"]["metric"] = "acc_norm"
    with pytest.raises(ValueError, match="ARC/HellaSwag"):
        validate_plan(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold", True),
        ("choices", ["a", ""]),
        ("metric", "acc"),
        ("normalization_lengths", [1, 5]),
    ],
)
def test_manifest_gold_choices_and_normalization_rejected(frozen, field, value):
    plan, manifest, _ = frozen
    manifest["items"][0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(plan, manifest)


def test_rounding_tolerance_cannot_change_first_argmax_tie(frozen):
    plan, manifest, runs = frozen
    model = plan["models"][0]
    run = runs[model["key"]]
    run["rows"][0].update(
        choice_log_likelihoods=[-1, -4],
        choice_scores=[-1, -1 + 1e-13],
        prediction=1,
        correct=False,
    )
    with pytest.raises(ValueError, match="first score argmax"):
        validate_run(
            run,
            model,
            plan,
            validate_manifest(plan, manifest),
            PLAN_HASH,
            ITEMS_HASH,
        )


def test_incomplete_raw_file_never_publishes_partial_summary(tmp_path, frozen):
    results, plan, items = write_artifacts(tmp_path, frozen)
    path = results / "qwen7.json.gz"
    run = json.loads(gzip.decompress(path.read_bytes()))
    run["rows"].pop()
    path.write_bytes(gzip.compress(json.dumps(run).encode(), mtime=0))
    with pytest.raises(ValueError, match="coverage"):
        build_report(results, plan, items)
    assert not (results / "summary.json").exists()
    assert not (results / "paired-change.svg").exists()
