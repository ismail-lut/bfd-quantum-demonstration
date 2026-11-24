"""Summary statistics, figure and tables.

Cost and gap statistics are computed over FEASIBLE runs only: a run that
returned no feasible solution has no meaningful cost (an invalid
configuration can cost less than the optimum). Feasible-read share is
computed over ALL runs, since a run with zero valid reads is itself data.

Averaging cost over all runs instead would report, for example, EUR 5,705
for the QPU at penalty 0.1 on instance B: below the optimum of EUR 8,253,
which no valid solution can be.
"""

import numpy as np
import pandas as pd

SOLVER_LABEL = {"sa": "Simulated annealing (classical)",
                "qpu": "D-Wave Advantage QPU",
                "qpu_fixed_chain": "QPU, fixed chain strength",
                "hybrid": "Leap hybrid solver"}
STYLE = {"sa":              dict(color="#4C72B0", marker="o", linestyle="--"),
         "qpu":             dict(color="#C44E52", marker="s", linestyle="-"),
         "qpu_fixed_chain": dict(color="#8172B2", marker="D", linestyle="-."),
         "hybrid":          dict(color="#55A868", marker="^", linestyle=":")}
ORDER = ["sa", "qpu", "qpu_fixed_chain", "hybrid"]


def load_results(path):
    """Read a results CSV, normalizing the `feasible` column to bool."""
    df = pd.read_csv(path)
    if df["feasible"].dtype != bool:
        df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"])
    return df


def aggregate(df):
    """One row per (instance, solver, penalty_scale)."""
    out = []
    for (inst, solver, ps), s in df.groupby(["instance", "solver", "penalty_scale"]):
        f = s[s.feasible]
        out.append({
            "instance": inst, "solver": solver, "penalty_scale": ps,
            "n_runs": len(s), "n_feasible_runs": len(f),
            "mean_cost": f.best_cost.mean() if len(f) else np.nan,
            "mean_gap": f.optimality_gap_pct.mean() if len(f) else np.nan,
            "sd_gap": f.optimality_gap_pct.std() if len(f) > 1 else np.nan,
            "min_gap": f.optimality_gap_pct.min() if len(f) else np.nan,
            "max_gap": f.optimality_gap_pct.max() if len(f) else np.nan,
            "feasible_fraction": s.feasible_fraction.mean(),
            "chain_break": s.chain_break_fraction.mean(),
            "qpu_us": s.qpu_access_time_us.mean(),
            "physical_qubits": s.physical_qubits.mean(),
            "max_chain": s.max_chain_length.max()})
    return pd.DataFrame(out)


def figure_1(df, instance="B", path="figure1_penalty_sensitivity.png"):
    """Optimality gap and feasible-read share against penalty scale."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    g = aggregate(df)
    g = g[g.instance == instance]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for solver in ORDER:
        s = g[g.solver == solver].sort_values("penalty_scale")
        if s.empty:
            continue
        st = STYLE[solver]
        ok = s.dropna(subset=["mean_gap"])
        rel = ok[ok.n_feasible_runs >= 0.5 * ok.n_runs]
        unrel = ok[ok.n_feasible_runs < 0.5 * ok.n_runs]
        ax1.plot(rel.penalty_scale, rel.mean_gap, label=SOLVER_LABEL[solver],
                 linewidth=1.8, markersize=6, **st)
        if len(unrel):
            ax1.plot(unrel.penalty_scale, unrel.mean_gap, linestyle="none",
                     marker=st["marker"], markersize=6, markerfacecolor="white",
                     markeredgecolor=st["color"], markeredgewidth=1.4)
        if ok.sd_gap.notna().any():
            ax1.fill_between(rel.penalty_scale,
                             rel.mean_gap - rel.sd_gap.fillna(0),
                             rel.mean_gap + rel.sd_gap.fillna(0),
                             color=st["color"], alpha=0.14, linewidth=0)
        dead = s[s.n_feasible_runs == 0]
        if len(dead):
            j = 1.0 + 0.055 * (ORDER.index(solver) - 1.5)  # keep series visible
            ax1.plot(dead.penalty_scale * j, np.zeros(len(dead)), linestyle="none",
                     marker="x", markersize=8, markeredgewidth=1.8,
                     color=st["color"], clip_on=False)
        ax2.plot(s.penalty_scale, s.feasible_fraction * 100,
                 linewidth=1.8, markersize=6, **st)

    ax1.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax1.annotate("Classical optimum (0% gap)", xy=(0.985, 0.0),
                 xycoords=("axes fraction", "data"), xytext=(0, 6),
                 textcoords="offset points", fontsize=8, style="italic",
                 ha="right", va="bottom", color="#333333")
    n_vars = int(df[df.instance == instance].n_variables.iloc[0])
    ax1.set_ylabel("Optimality gap (%)")
    ax1.set_title(f"Solution quality against penalty weight — instance {instance} "
                  f"({n_vars} variables)", fontsize=11, pad=10)
    ax1.grid(alpha=0.25, linewidth=0.6)
    h, lab = ax1.get_legend_handles_labels()
    h += [Line2D([], [], linestyle="none", marker="x", color="#555555",
                 markersize=8, markeredgewidth=1.8),
          Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                 markeredgecolor="#555555", markeredgewidth=1.4, markersize=6)]
    lab += ["no feasible solution in any run", "feasible in a minority of runs"]
    ax1.legend(h, lab, frameon=False, fontsize=8.5, loc="upper left",
               handletextpad=0.8)
    ax2.set_ylabel("Feasible reads (%)")
    ax2.set_xlabel("Penalty scale")
    ax2.set_xscale("log")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ticks = sorted(g.penalty_scale.unique())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t:g}" for t in ticks])
    ax2.minorticks_off()
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def table_1(df, instance="B"):
    """Per-penalty detail for one instance."""
    g = aggregate(df)
    g = g[g.instance == instance].copy()
    out = pd.DataFrame({
        "Solver": g.solver.map(SOLVER_LABEL),
        "Penalty": g.penalty_scale,
        "Runs": g.n_runs, "Feasible runs": g.n_feasible_runs,
        "Mean cost (EUR)": g.mean_cost.round(0),
        "Mean gap (%)": g.mean_gap.round(1), "SD (pp)": g.sd_gap.round(1),
        "Best (%)": g.min_gap.round(1), "Worst (%)": g.max_gap.round(1),
        "Feasible reads (%)": (g.feasible_fraction * 100).round(0),
        "Chain breaks (%)": (g.chain_break * 100).round(1),
        "QPU time (ms)": (g.qpu_us / 1000).round(1)})
    return out.sort_values(["Solver", "Penalty"]).reset_index(drop=True)


def table_2(df):
    """Best and worst penalty setting per instance and solver."""
    g = aggregate(df).dropna(subset=["mean_gap"])
    rows = []
    for (i, s), x in g.groupby(["instance", "solver"]):
        b, w = x.loc[x.mean_gap.idxmin()], x.loc[x.mean_gap.idxmax()]
        rows.append({"Instance": i,
                     "Variables": int(df[df.instance == i].n_variables.iloc[0]),
                     "Solver": SOLVER_LABEL[s], "Replications": int(x.n_runs.max()),
                     "Best penalty": b.penalty_scale,
                     "Best gap (%)": round(b.mean_gap, 1),
                     "Worst penalty": w.penalty_scale,
                     "Worst gap (%)": round(w.mean_gap, 1),
                     "Spread (pp)": round(w.mean_gap - b.mean_gap, 1)})
    return pd.DataFrame(rows).sort_values(["Instance", "Solver"]).reset_index(drop=True)


def table_embedding(df):
    """Minor-embedding overhead by instance, over all QPU runs."""
    q = df[df.solver.str.startswith("qpu")]
    e = q.groupby("instance").agg(
        logical_variables=("n_variables", "first"),
        physical_qubits_mean=("physical_qubits", "mean"),
        longest_chain=("max_chain_length", "max"),
        chain_break_fraction_mean=("chain_break_fraction", "mean"),
        qpu_runs=("solver", "size"))
    e["qubits_per_variable"] = e.physical_qubits_mean / e.logical_variables
    return e.round(2).reset_index()


def table_usage(df):
    """Solver calls and recorded solver time, by solver."""
    rows = []
    for solver, s in df.groupby("solver"):
        rows.append({"solver": solver, "calls": len(s),
                     "qpu_access_time_s": round(s.qpu_access_time_us.sum() / 1e6, 3),
                     "charge_time_s": round(s.charge_time_us.sum() / 1e6, 3)})
    return pd.DataFrame(rows)


def chain_break_summary(df):
    """Chain breaks by instance and solver, in percent of chains per read.

    chain_break_fraction is the share of chains broken in a read, averaged
    over the reads of one submission. Report it unrounded: values here are
    of order 0.01 %, which rounds to 0.0 % but is not zero.
    """
    q = df[df.solver.str.startswith("qpu")]
    out = q.groupby(["instance", "solver"]).chain_break_fraction.agg(
        submissions="size",
        submissions_with_breaks=lambda x: int((x > 0).sum()),
        mean_pct=lambda x: x.mean() * 100,
        max_pct=lambda x: x.max() * 100)
    return out.round(4).reset_index()
