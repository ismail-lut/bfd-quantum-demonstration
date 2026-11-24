"""Regenerate every figure and table from the archived results.

Usage:  python scripts/reproduce_analysis.py [path/to/bfd_results.csv]

Needs no D-Wave account and no solver time. Defaults to the archived run of
5 November 2026 in data/results/bfd_results.csv and writes to results/.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from bfd.analysis import (figure_1, load_results, chain_break_summary,  # noqa: E402
                          table_1, table_2, table_embedding, table_usage)
from bfd.instances import build_instance  # noqa: E402
from bfd.maps import figure_instance_map  # noqa: E402
from bfd.solvers import solve_exact  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "results" / "bfd_results.csv"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df = load_results(src)
    print(f"read {len(df)} runs from {src}")

    figure_1(df, "B", out / "figure1_penalty_sensitivity.png")
    inst = build_instance("B")
    figure_instance_map(inst, solve_exact(inst), out / "figure2_instance_map_B.png")
    table_1(df, "B").to_csv(out / "table1_instance_B.csv", index=False)
    table_2(df).to_csv(out / "table2_robustness.csv", index=False)
    table_embedding(df).to_csv(out / "table3_embedding.csv", index=False)
    table_usage(df).to_csv(out / "table4_solver_usage.csv", index=False)
    chain_break_summary(df).to_csv(out / "table5_chain_breaks.csv", index=False)
    for p in sorted(out.iterdir()):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
