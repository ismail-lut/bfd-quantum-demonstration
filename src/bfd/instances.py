"""Problem instances: reverse logistics sorting facility location.

A company collects returned products at m collection points and must decide
which of n candidate sorting facilities to open, and which facility each
collection point sends its returns to, minimizing facility opening cost plus
transport cost.

Decision variables
    y_j  = 1 if facility j is opened
    x_ij = 1 if collection point i is assigned to facility j

Instances are generated from a fixed seed, so they are identical on every
run. The instance generator must not be changed: the reference optima
(EUR 7,571 / 8,253 / 14,154) and the archived results depend on it.
"""

import numpy as np

# name: (collection points m, candidate facilities n)
INSTANCES = {"A": (8, 3), "B": (12, 4), "C": (20, 5)}
SEED = 42
COST_PER_UNIT_KM = 0.15  # EUR per returned unit per km


def build_instance(name, seed=SEED):
    """Generate instance `name` ('A', 'B' or 'C').

    Collection points lie on a 100 x 100 km grid, candidate facilities in its
    central 60 x 60 km. Transport cost is proportional to Euclidean distance
    and return volume.
    """
    m, n = INSTANCES[name]
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, size=(m, 2))
    facilities = rng.uniform(20, 80, size=(n, 2))
    volumes = rng.integers(50, 200, size=m)               # units per period
    open_cost = rng.integers(800, 2000, size=n)           # EUR per period
    dist = np.linalg.norm(points[:, None, :] - facilities[None, :, :], axis=2)
    transport = dist * volumes[:, None] * COST_PER_UNIT_KM
    return {"name": name, "m": m, "n": n, "transport": transport,
            "open_cost": open_cost, "volumes": volumes,
            "points": points, "facilities": facilities}


def n_variables(inst):
    """Number of binary variables: m*n assignments plus n facility switches."""
    return inst["m"] * inst["n"] + inst["n"]


def decode(sample, inst):
    """Turn switch settings into a business answer and check both constraints.

    Every solver in this study, classical or quantum, returns switch settings.
    Because the QUBO enforces constraints by penalty rather than by rule, a
    sample can score well and still be invalid, so feasibility is checked here
    rather than assumed. Infeasible samples are never counted as results.
    """
    m, n = inst["m"], inst["n"]
    assign, violations = {}, []
    for i in range(m):
        chosen = [j for j in range(n) if sample.get(f"x_{i}_{j}", 0) == 1]
        if len(chosen) != 1:
            violations.append(f"point {i} -> {len(chosen)} facilities")
        if chosen:
            assign[i] = chosen[0]
    opened = {j for j in range(n) if sample.get(f"y_{j}", 0) == 1}
    for i, j in assign.items():
        if j not in opened:
            violations.append(f"point {i} -> closed facility {j}")
    cost = sum(inst["open_cost"][j] for j in opened)
    cost += sum(inst["transport"][i][j] for i, j in assign.items())
    return {"assignment": assign, "opened": sorted(opened), "cost": float(cost),
            "feasible": len(violations) == 0, "violations": violations}


def instance_to_dict(inst, optimum=None):
    """Serializable form of an instance, for data/instances/*.json."""
    out = {
        "name": inst["name"],
        "seed": SEED,
        "collection_points": int(inst["m"]),
        "candidate_facilities": int(inst["n"]),
        "binary_variables": n_variables(inst),
        "cost_per_unit_km_eur": COST_PER_UNIT_KM,
        "points_km": np.round(inst["points"], 6).tolist(),
        "facilities_km": np.round(inst["facilities"], 6).tolist(),
        "return_volumes_units": inst["volumes"].astype(int).tolist(),
        "facility_open_cost_eur": inst["open_cost"].astype(int).tolist(),
        "transport_cost_eur": np.round(inst["transport"], 6).tolist(),
    }
    if optimum is not None:
        out["optimum"] = {
            "cost_eur": round(optimum["cost"], 4),
            "facilities_opened": [int(j) for j in optimum["opened"]],
            "assignment": {str(i): int(j) for i, j in optimum["assignment"].items()},
            "solver": "OR-Tools CP-SAT (proven optimal)",
        }
    return out
