"""Write the three problem instances and their proven optima to data/instances/.

Usage:  python scripts/export_instances.py

The JSON files are a convenience for readers who want the data without
running Python. The generator in src/bfd/instances.py remains the source of
truth; tests/test_reproducibility.py checks the two agree.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from bfd.instances import INSTANCES, build_instance, instance_to_dict, n_variables  # noqa: E402
from bfd.solvers import solve_bruteforce, solve_exact  # noqa: E402


def main():
    out_dir = ROOT / "data" / "instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in INSTANCES:
        inst = build_instance(name)
        opt = solve_exact(inst)
        brute = solve_bruteforce(inst)
        assert abs(opt["cost"] - brute["cost"]) < 1e-6, f"{name}: CP-SAT and brute force disagree"
        (out_dir / f"instance_{name}.json").write_text(
            json.dumps(instance_to_dict(inst, opt), indent=2) + "\n")
        summary.append({"instance": name,
                        "collection_points": inst["m"],
                        "candidate_facilities": inst["n"],
                        "binary_variables": n_variables(inst),
                        "optimum_eur": round(opt["cost"], 4),
                        "facilities_opened": " ".join(str(j) for j in opt["opened"]),
                        "total_return_volume_units": int(inst["volumes"].sum())})
        print(f"instance {name}: EUR {opt['cost']:,.4f}, open {opt['opened']}")
    pd.DataFrame(summary).to_csv(out_dir / "instances_summary.csv", index=False)
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
