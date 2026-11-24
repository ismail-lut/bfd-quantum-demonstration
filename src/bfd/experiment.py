"""Experimental procedure: one solver call in, one schema row out.

Row schema (identical to data/results/bfd_results.csv)
    instance, n_variables, solver, penalty_scale, penalty_A, penalty_B,
    replication, chip_id, num_reads, annealing_time_us, chain_strength,
    chain_break_fraction, max_chain_length, physical_qubits,
    qpu_access_time_us, charge_time_us, wall_seconds, best_cost, feasible,
    optimality_gap_pct, feasible_fraction

Solver labels: "sa" (simulated annealing), "qpu" (Advantage QPU, automatic
chain strength), "qpu_fixed_chain" (QPU, chain strength held constant),
"hybrid" (Leap hybrid BQM solver).
"""

import time

import numpy as np
import pandas as pd

from .instances import n_variables
from .qubo import build_bqm
from .solvers import summarize

PENALTIES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
REPLICATIONS = 20
HYBRID_REPLICATIONS = 3
NUM_READS = 1000
ANNEALING_TIME_US = 20


def run_one(inst, optimum, solver, penalty_scale, replication,
            num_reads=NUM_READS, annealing_time=ANNEALING_TIME_US,
            chain_strength=None, qpu_sampler=None, hybrid_sampler=None,
            chip_id="", seed=None):
    """Run one solver call and return one schema row (a dict)."""
    bqm, w = build_bqm(inst, penalty_scale)
    row = {"instance": inst["name"], "n_variables": n_variables(inst),
           "solver": solver, "penalty_scale": penalty_scale,
           "penalty_A": w["A"], "penalty_B": w["B"], "replication": replication,
           "chip_id": "", "num_reads": np.nan, "annealing_time_us": np.nan,
           "chain_strength": np.nan, "chain_break_fraction": np.nan,
           "max_chain_length": np.nan, "physical_qubits": np.nan,
           "qpu_access_time_us": np.nan, "charge_time_us": np.nan}
    label = f"BFD-{inst['name']}-p{penalty_scale}-r{replication}"

    t0 = time.perf_counter()
    if solver == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=seed)
        row["num_reads"] = num_reads
    elif solver in ("qpu", "qpu_fixed_chain"):
        kw = dict(num_reads=num_reads, annealing_time=annealing_time,
                  label=label, return_embedding=True)
        if chain_strength is not None:
            kw["chain_strength"] = chain_strength
        ss = qpu_sampler.sample(bqm, **kw)
        row["num_reads"] = num_reads
        row["annealing_time_us"] = annealing_time
        row["chip_id"] = chip_id
    elif solver == "hybrid":
        ss = hybrid_sampler.sample(bqm, label=label)
    else:
        raise ValueError(f"unknown solver {solver!r}")
    row["wall_seconds"] = time.perf_counter() - t0

    best, fr, tr = summarize(ss, inst)
    row["best_cost"] = best["cost"]
    row["feasible"] = best["feasible"]
    row["optimality_gap_pct"] = ((best["cost"] - optimum) / optimum * 100
                                 if best["feasible"] else np.nan)
    row["feasible_fraction"] = fr / tr if tr else 0.0

    # Timing: the QPU nests it under "timing"; the hybrid solver reports it at
    # the top level of sampleset.info.
    info = ss.info or {}
    timing = info.get("timing", {}) or {}
    row["qpu_access_time_us"] = timing.get("qpu_access_time",
                                           info.get("qpu_access_time", np.nan))
    row["charge_time_us"] = info.get("charge_time",
                                     timing.get("charge_time", np.nan))

    if solver.startswith("qpu"):
        ectx = info.get("embedding_context", {}) or {}
        emb = ectx.get("embedding", {})
        if emb:
            chains = [len(v) for v in emb.values()]
            row["physical_qubits"] = sum(chains)
            row["max_chain_length"] = max(chains)
        row["chain_strength"] = ectx.get("chain_strength", np.nan)
        if "chain_break_fraction" in ss.record.dtype.names:
            row["chain_break_fraction"] = float(np.average(
                ss.record.chain_break_fraction, weights=ss.record.num_occurrences))
    return row


def run_sweep(inst, optimum, solver, penalties=PENALTIES,
              replications=REPLICATIONS, progress=False, **kw):
    """All penalty settings x replications for one instance and solver."""
    rows = []
    for ps in penalties:
        for rep in range(replications):
            # simulated annealing gets a distinct, reproducible seed per cell
            seed = 1000 * rep + int(ps * 100) if solver == "sa" else None
            rows.append(run_one(inst, optimum, solver, ps, rep, seed=seed, **kw))
        if progress:
            gaps = [r["optimality_gap_pct"] for r in rows[-replications:]]
            ok = [g for g in gaps if not np.isnan(g)]
            msg = (f"mean gap {np.mean(ok):5.1f}%  ({len(ok)}/{replications} feasible)"
                   if ok else "no feasible solution")
            print(f"  instance {inst['name']}  {solver:<15} penalty {ps:<5} -> {msg}")
    return pd.DataFrame(rows)
