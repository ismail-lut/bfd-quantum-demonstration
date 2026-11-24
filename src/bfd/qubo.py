"""QUBO formulation of the facility location problem.

A QUBO cannot state a rule that must hold, so both constraints become
penalties added to the cost:

    (1) each point assigned exactly once:   A * (sum_j x_ij - 1)^2
    (2) no assignment to a closed facility: B * x_ij * (1 - y_j)

Both expand into linear and pairwise terms using x^2 = x for binaries.
`penalty_scale` multiplies both weights and is the single factor varied in
the experiment. Too low and the sampler breaks constraints to save cost;
too high and the penalties swamp the cost signal.
"""

import itertools


def build_qubo(inst, penalty_scale=2.0):
    """Return (Q, offset, weights) for instance `inst`.

    A is scaled off the largest transport cost times n; B off the larger of
    the largest transport cost and the largest opening cost.
    """
    m, n = inst["m"], inst["n"]
    transport, open_cost = inst["transport"], inst["open_cost"]
    A = penalty_scale * float(transport.max()) * n
    B = penalty_scale * float(max(transport.max(), open_cost.max()))
    Q = {}

    def add(u, v, w):
        key = (u, v) if u <= v else (v, u)
        Q[key] = Q.get(key, 0.0) + w

    def x(i, j):
        return f"x_{i}_{j}"

    def y(j):
        return f"y_{j}"

    # objective
    for j in range(n):
        add(y(j), y(j), float(open_cost[j]))
    for i in range(m):
        for j in range(n):
            add(x(i, j), x(i, j), float(transport[i][j]))

    # constraint (1): one-hot assignment
    offset = 0.0
    for i in range(m):
        for j in range(n):
            add(x(i, j), x(i, j), -A)
        for j, k in itertools.combinations(range(n), 2):
            add(x(i, j), x(i, k), 2 * A)
        offset += A

    # constraint (2): assignment implies facility open
    for i in range(m):
        for j in range(n):
            add(x(i, j), x(i, j), B)
            add(x(i, j), y(j), -B)

    return Q, offset, {"A": A, "B": B}


def build_bqm(inst, penalty_scale):
    """The QUBO as a dimod BinaryQuadraticModel, plus its penalty weights."""
    import dimod
    Q, offset, weights = build_qubo(inst, penalty_scale=penalty_scale)
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset), weights
