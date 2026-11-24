"""Exact classical reference and helpers shared by all samplers."""

import time

import numpy as np

from .instances import decode


def solve_exact(inst):
    """Proven optimum via OR-Tools CP-SAT.

    CP-SAT works in integers, so costs are rounded to whole cents for the
    solve. The reported cost is recomputed in euros from the unrounded data.
    For instances A, B and C the result matches an exhaustive search over all
    facility subsets in full floating-point precision (see solve_bruteforce
    and tests/test_reproducibility.py), so the rounding does not change the
    optimum here.
    """
    from ortools.sat.python import cp_model
    m, n = inst["m"], inst["n"]
    scale = 100
    transport = np.rint(inst["transport"] * scale).astype(int)
    open_cost = np.rint(np.asarray(inst["open_cost"], dtype=float) * scale).astype(int)

    model = cp_model.CpModel()
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(m) for j in range(n)}
    y = {j: model.NewBoolVar(f"y_{j}") for j in range(n)}
    for i in range(m):
        model.AddExactlyOne(x[(i, j)] for j in range(n))
        for j in range(n):
            model.Add(x[(i, j)] <= y[j])
    model.Minimize(sum(int(open_cost[j]) * y[j] for j in range(n))
                   + sum(int(transport[i][j]) * x[(i, j)]
                         for i in range(m) for j in range(n)))

    solver = cp_model.CpSolver()
    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT did not prove optimality (status {status})")

    sample = {f"x_{i}_{j}": solver.Value(x[(i, j)]) for i in range(m) for j in range(n)}
    sample.update({f"y_{j}": solver.Value(y[j]) for j in range(n)})
    result = decode(sample, inst)
    result["wall_seconds"] = elapsed
    return result


def solve_bruteforce(inst):
    """Exact optimum by enumerating every non-empty set of open facilities.

    Valid because the problem is uncapacitated: once the open set is fixed,
    each point goes to its cheapest open facility. Feasible only for small n
    (2^n - 1 subsets); used as an independent check on CP-SAT.
    """
    import itertools
    n = inst["n"]
    T, F = inst["transport"], inst["open_cost"]
    best_cost, best_set = None, None
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            cost = float(sum(F[j] for j in S) + T[:, list(S)].min(axis=1).sum())
            if best_cost is None or cost < best_cost:
                best_cost, best_set = cost, list(S)
    return {"cost": best_cost, "opened": best_set}


def summarize(sampleset, inst):
    """Best feasible solution in a sampleset, and the feasible-read count.

    Returns (best, feasible_reads, total_reads). If no read is feasible,
    `best` is the lowest-energy read, marked infeasible.
    """
    best, feasible_reads, total_reads = None, 0, 0
    for rec in sampleset.data(["sample", "energy", "num_occurrences"]):
        d = decode(rec.sample, inst)
        total_reads += rec.num_occurrences
        if d["feasible"]:
            feasible_reads += rec.num_occurrences
            if best is None or d["cost"] < best["cost"]:
                best = d
    if best is None:
        best = decode(sampleset.first.sample, inst)
    return best, feasible_reads, total_reads
