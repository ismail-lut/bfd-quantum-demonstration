"""Reproducibility checks.  Run:  pytest -q

None of these tests needs a D-Wave account. Hardware code paths are
exercised with mock samplers that return sampleset metadata in the format
D-Wave Leap returns it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfd import INSTANCES, build_bqm, build_instance, n_variables  # noqa: E402
from bfd.analysis import aggregate, load_results, table_2  # noqa: E402
from bfd.experiment import run_one, run_sweep  # noqa: E402
from bfd.solvers import solve_bruteforce, solve_exact, summarize  # noqa: E402

ARCHIVE = ROOT / "data" / "results" / "bfd_results.csv"

# Proven optima (CP-SAT, confirmed by exhaustive search).
OPTIMA_EUR = {"A": 7571.3243, "B": 8253.4102, "C": 14153.8623}
OPENED = {"A": [1], "B": [0, 3], "C": [2, 3]}


# ---------------------------------------------------------------- instances
@pytest.mark.parametrize("name", list(INSTANCES))
def test_optimum_matches_reference(name):
    r = solve_exact(build_instance(name))
    assert r["cost"] == pytest.approx(OPTIMA_EUR[name], abs=1e-3)
    assert r["opened"] == OPENED[name]
    assert r["feasible"]


@pytest.mark.parametrize("name", list(INSTANCES))
def test_cpsat_agrees_with_exhaustive_search(name):
    inst = build_instance(name)
    assert solve_exact(inst)["cost"] == pytest.approx(solve_bruteforce(inst)["cost"], abs=1e-6)


@pytest.mark.parametrize("name", list(INSTANCES))
def test_instance_json_matches_generator(name):
    data = json.loads((ROOT / "data" / "instances" / f"instance_{name}.json").read_text())
    inst = build_instance(name)
    assert data["binary_variables"] == n_variables(inst)
    np.testing.assert_allclose(data["transport_cost_eur"], inst["transport"], atol=1e-6)
    assert data["facility_open_cost_eur"] == inst["open_cost"].tolist()
    assert data["optimum"]["cost_eur"] == pytest.approx(OPTIMA_EUR[name], abs=1e-3)


# ---------------------------------------------------------------- QUBO
def test_qubo_reproduces_optimum_on_instance_A():
    from dwave.samplers import SimulatedAnnealingSampler
    inst = build_instance("A")
    bqm, _ = build_bqm(inst, 0.25)
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=1000, seed=42)
    best, _, _ = summarize(ss, inst)
    assert best["cost"] == pytest.approx(OPTIMA_EUR["A"], abs=1e-3)


def test_qubo_energy_equals_cost_for_feasible_solution():
    """For a feasible assignment every penalty is zero, so energy == cost."""
    inst = build_instance("B")
    bqm, _ = build_bqm(inst, 1.0)
    opt = solve_exact(inst)
    sample = {v: 0 for v in bqm.variables}
    for i, j in opt["assignment"].items():
        sample[f"x_{i}_{j}"] = 1
    for j in opt["opened"]:
        sample[f"y_{j}"] = 1
    assert bqm.energy(sample) == pytest.approx(opt["cost"], abs=1e-6)


def test_fixed_chain_strength_matches_archived_run():
    from dwave.embedding.chain_strength import uniform_torque_compensation
    bqm, _ = build_bqm(build_instance("B"), 1.0)
    assert round(float(uniform_torque_compensation(bqm))) == 45843


# ---------------------------------------------------------------- hardware code paths (mocked)
class _MockQPU:
    """Stands in for EmbeddingComposite(DWaveSampler()): QPU-style metadata."""

    def sample(self, bqm, **kw):
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=50, seed=1)
        ss = ss.aggregate()
        cbf = np.zeros(len(ss))
        import dimod
        ss = dimod.SampleSet.from_samples_bqm(
            (ss.record.sample, ss.variables), bqm,
            num_occurrences=ss.record.num_occurrences, chain_break_fraction=cbf)
        ss.info.update({
            "timing": {"qpu_access_time": 126000.0},
            "embedding_context": {"embedding": {v: (0, 1) for v in bqm.variables},
                                  "chain_strength": kw.get("chain_strength", 999.0)}})
        return ss


class _MockHybrid:
    """Stands in for LeapHybridSampler: timing at the top level of info."""

    def sample(self, bqm, **kw):
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=50, seed=1)
        ss.info.update({"qpu_access_time": 40000, "charge_time": 2991000, "run_time": 2991000})
        return ss


def test_qpu_row_records_embedding_timing_and_chip():
    inst = build_instance("A")
    row = run_one(inst, OPTIMA_EUR["A"], "qpu", 0.25, 0,
                  qpu_sampler=_MockQPU(), chip_id="Advantage_system6")
    assert row["chip_id"] == "Advantage_system6"
    assert row["qpu_access_time_us"] == 126000.0
    assert row["physical_qubits"] == 2 * n_variables(inst)
    assert row["max_chain_length"] == 2
    assert row["chain_break_fraction"] == 0.0
    assert row["num_reads"] == 1000 and row["annealing_time_us"] == 20


def test_fixed_chain_strength_is_passed_through():
    row = run_one(build_instance("A"), OPTIMA_EUR["A"], "qpu_fixed_chain", 1.0, 0,
                  qpu_sampler=_MockQPU(), chain_strength=45843.0)
    assert row["chain_strength"] == 45843.0


def test_hybrid_timing_is_read_from_top_level_info():
    """The hybrid solver reports timing at the top level of sampleset.info,
    not under "timing" as the QPU does; reading the wrong place records blanks."""
    row = run_one(build_instance("A"), OPTIMA_EUR["A"], "hybrid", 0.25, 0,
                  hybrid_sampler=_MockHybrid())
    assert row["charge_time_us"] == 2991000
    assert row["qpu_access_time_us"] == 40000


def test_sweep_schema_matches_archive():
    df = run_sweep(build_instance("A"), OPTIMA_EUR["A"], "sa",
                   penalties=[0.25], replications=2, num_reads=50)
    assert list(df.columns) == list(pd.read_csv(ARCHIVE, nrows=1).columns)


# ---------------------------------------------------------------- archived results
@pytest.fixture(scope="module")
def archive():
    return load_results(ARCHIVE)


def test_archive_shape(archive):
    assert len(archive) == 1236
    counts = archive.groupby("solver").size().to_dict()
    assert counts == {"hybrid": 36, "qpu": 720, "qpu_fixed_chain": 120, "sa": 360}
    assert set(archive.loc[archive.solver.str.startswith("qpu"), "chip_id"]) == {"Advantage_system6"}


def test_archive_reproduces_reported_figures(archive):
    g = aggregate(archive).set_index(["instance", "solver", "penalty_scale"])
    qpu_b = g.loc[("B", "qpu")]
    assert round(qpu_b.loc[0.25, "mean_gap"], 1) == 9.3
    assert round(qpu_b.loc[4.0, "mean_gap"], 1) == 30.6
    assert qpu_b.loc[0.1, "n_feasible_runs"] == 8
    assert round(qpu_b.loc[0.1, "mean_cost"]) == 9571       # feasible-only mean
    assert round(g.loc[("B", "hybrid", 0.25), "mean_gap"], 1) == 0.0
    fixed = g.loc[("B", "qpu_fixed_chain")]
    assert fixed.loc[0.25, "n_feasible_runs"] == 0
    assert round(fixed.loc[0.5, "mean_gap"], 1) == 78.2


def test_spread_figures(archive):
    t = table_2(archive).set_index(["Instance", "Solver"])["Spread (pp)"]
    assert t[("B", "D-Wave Advantage QPU")] == 21.4
    assert t[("B", "Simulated annealing (classical)")] == 21.5
    assert t[("B", "QPU, fixed chain strength")] == 50.3


def test_chain_breaks_rare_but_present(archive):
    """Chain breaks were rare, not absent: of order 0.01 %, which rounds to
    0.0 % at one decimal place but is not zero."""
    q = archive[archive.solver.str.startswith("qpu")]
    assert (q.chain_break_fraction > 0).sum() == 667
    assert q.chain_break_fraction.mean() * 100 == pytest.approx(0.0115, abs=5e-4)
    assert q.chain_break_fraction.max() * 100 == pytest.approx(0.3276, abs=5e-4)


def test_recorded_solver_time(archive):
    q = archive[archive.solver.str.startswith("qpu")]
    assert q.qpu_access_time_us.sum() / 1e6 == pytest.approx(106.1, abs=0.05)
    h = archive[archive.solver == "hybrid"]
    assert h.charge_time_us.sum() / 1e6 == pytest.approx(107.686, abs=0.001)


# ---------------------------------------------------------------- notebook
def test_no_credentials_anywhere():
    import re
    pat = re.compile(r"(tvcp|DEV)-[A-Za-z0-9-]{8,}")
    for f in ROOT.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".ipynb", ".md", ".txt", ".csv", ".json", ".cff"}:
            assert not pat.search(f.read_text(errors="ignore")), f"possible API token in {f}"


def test_notebook_is_in_sync_with_package():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "BFD_quantum_demonstration.ipynb", as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for mod in ["instances", "solvers", "maps", "qubo", "experiment", "analysis"]:
        src = (ROOT / "src" / "bfd" / f"{mod}.py").read_text()
        body = "\n".join(l for l in src.splitlines() if not l.startswith("from ."))
        assert body.strip() in code, f"notebook out of date for {mod}.py: run scripts/build_notebook.py"
