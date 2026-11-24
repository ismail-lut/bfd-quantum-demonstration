"""Build notebooks/BFD_quantum_demonstration.ipynb from the package source.

Usage:  python scripts/build_notebook.py [--execute]

The notebook's code cells are generated from src/bfd/*.py, so the notebook
and the package cannot drift apart. Edit the package, then rebuild.
With --execute the notebook is run in reproduce mode (archived results, no
solver time) and saved with its outputs.
"""

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "bfd"
OUT = ROOT / "notebooks" / "BFD_quantum_demonstration.ipynb"


def module(name):
    """Module source with package-relative imports removed (the notebook
    defines every name in one namespace, in dependency order)."""
    text = (SRC / f"{name}.py").read_text()
    text = re.sub(r"^from \.\w+ import .*\n", "", text, flags=re.M)
    return f"# --- generated from src/bfd/{name}.py ---\n" + text.strip() + "\n"


cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))   # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))     # noqa: E731

# ------------------------------------------------------------------ title
md(r"""
# How much does the formulation matter?
## Penalty-weight sensitivity of a reverse logistics problem on D-Wave quantum annealing hardware

**Author:** Ismail Zahoor, Arcada University of Applied Sciences, 2026
**Hardware run:** 5 November 2026, D-Wave Advantage_system6 quantum processor and the Leap hybrid solver
`hybrid_binary_quadratic_model_version2p`

---

### What this notebook shows

Before a business problem can run on a quantum annealer, it has to be rewritten in a special mathematical form,
and the rewriting involves a choice the business problem does not dictate: how heavily to penalize breaking a
rule. This notebook measures **how much that one choice changes the quality of the answer**, holding the problem,
the solver and the hardware constant.

It walks through the whole process in eight steps, each building on the last:

| Step | What happens | Why the next step needs it |
|---|---|---|
| 1 | Configure | Chooses between reproducing the archived run and running a new one |
| 2 | Build the problem | A realistic recycling-logistics decision with fixed, reproducible data |
| 3 | Solve it exactly with a classical solver | The **best possible answer**, against which every later result is measured |
| 4 | Rewrite it for a quantum annealer | The only input format an annealer accepts; rules become penalties |
| 5 | Check the rewriting | Proves it is still the same problem before any solver time is spent |
| 6 | Connect to D-Wave and run one test | Confirms access, processor and recording for about one second of solver time |
| 7 | Run the experiment | Varies only the penalty weight, holding everything else constant |
| 8 | Figures and tables | The results |

**This is not a test of whether quantum computers beat classical ones.** The problems are deliberately small (27
to 105 yes/no decisions) so that a conventional solver finds the provably best answer in milliseconds. That is what
makes it possible to measure exactly how far from best every other answer is.

### Two ways to run it

- **Reproduce mode (default, `RUN_EXPERIMENT = False`).** Loads the archived results of 5 November 2026 and
  regenerates every figure and table. Needs no D-Wave account and uses no solver time. Every classical result,
  including the problems and their best answers, is recomputed exactly.
- **Run mode (`RUN_EXPERIMENT = True`).** Repeats the whole experiment on D-Wave Leap. Needs a Leap account and
  about two minutes of solver time; it takes about 35 minutes, mostly queueing. Quantum results will differ in
  detail from the archive, because the hardware is probabilistic and its calibration drifts, but the pattern
  across penalty weights should recur.

*The code cells are generated from the `src/bfd` package in the accompanying repository, so the notebook and the
package always match.*
""")

# ------------------------------------------------------------------ step 1
md(r"""
---
## Step 1 — Configure

Set `RUN_EXPERIMENT` here. Everything else in the notebook follows from it.

To run the experiment you need the **Solver API Token** from the D-Wave Leap dashboard
([cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap)), copied with its copy button. If your account has
several projects, select the right one first; tokens differ by project. Paste the **whole** token as
given; whatever it begins with is part of it. The safest place for it is Colab's Secrets panel (the key icon in
the left sidebar): add a secret named `DWAVE_API_TOKEN` and enable notebook access, so the token is never saved in
this file.
""")

code(r"""
RUN_EXPERIMENT = False          # True: run the full experiment on D-Wave (uses ~2 min of solver allowance)
PREFERRED_QPU = "Advantage_system6"   # the processor used on 5 November 2026
OUTPUT_DIR = "outputs"

import importlib.util, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules

# Install what Colab lacks. If Colab then asks to restart the session, restart and run again from the top.
missing = [p for p, mod in [("dwave-ocean-sdk", "dimod"), ("ortools", "ortools")]
           if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])

def load_token():
    token = os.environ.get("DWAVE_API_TOKEN")
    if not token and IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("DWAVE_API_TOKEN")
        except Exception:
            token = None
    return token

if RUN_EXPERIMENT:
    token = load_token()
    if not token:
        raise RuntimeError("RUN_EXPERIMENT is True but no DWAVE_API_TOKEN was found "
                           "(environment variable or Colab secret).")
    os.environ["DWAVE_API_TOKEN"] = token.strip()
    print(f"Token loaded (length {len(token.strip())}).")

os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
from IPython.display import Image, display
import dimod, ortools
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
print(f"Mode: {'RUN EXPERIMENT on D-Wave' if RUN_EXPERIMENT else 'REPRODUCE from archived results'}")
print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"dimod {dimod.__version__} | ortools {ortools.__version__}")
""")

# ------------------------------------------------------------------ step 2
md(r"""
---
## Step 2 — The problem

In a **circular supply chain**, used products flow back from customers to be repaired, refurbished or recycled,
instead of ending as waste. Organizing those return flows efficiently is a classic optimization problem.

**The decision.** A company collects returned products at a number of collection points. It must decide which
sorting facilities to open, and which collection point sends its returns to which facility. Opening a facility
has a fixed cost; transport cost grows with distance and return volume. Too few facilities raise transport
cost; too many raise fixed cost.

This is *facility location* for returns, also called reverse logistics network design. It was chosen over
vehicle routing because it needs far fewer yes/no variables (about `m × n + n` rather than `n²`), and because small
versions can be solved exactly, giving a known best answer to measure against.

**Decision variables (on/off switches)**
- `y_j = 1` if facility *j* is opened
- `x_ij = 1` if collection point *i* is assigned to facility *j*

**Rules that must hold**
1. Every collection point is assigned to exactly one facility
2. No collection point is assigned to a facility that is not open

| Instance | Collection points | Facilities | Binary variables | Role |
|---|---|---|---|---|
| A | 8 | 3 | 27 | Smaller check |
| B | 12 | 4 | 52 | **Main test case** |
| C | 20 | 5 | 105 | Larger check, near the limit of what fits comfortably on the processor |

The seed is fixed at 42, so the instances are identical on every run. `decode` turns any solver's switch settings
back into a business answer and checks both rules: because rules become penalties in Step 4, a solver can return a
low-scoring answer that breaks them, and such answers are never counted as results.
""")

code(module("instances"))

code(r"""
pd.DataFrame([{"Instance": k, "Collection points": m, "Facilities": n, "Binary variables": m * n + n}
              for k, (m, n) in INSTANCES.items()])
""")

# ------------------------------------------------------------------ step 3
md(r"""
---
## Step 3 — The exact classical optimum

Every later result is reported as an **optimality gap**: the percentage by which a solution's cost exceeds the
true minimum. Without the true minimum, quantum results could only be compared with each other.

OR-Tools CP-SAT, a widely used open-source solver, returns a proven optimum in milliseconds. Because facilities
here have no capacity limit, the optimum can also be found by trying every set of open facilities and sending each
point to its cheapest open one;
`solve_bruteforce` does this in full floating-point precision as an independent check on CP-SAT.

**Expected:** EUR 7,571 (A), EUR 8,253 (B), EUR 14,154 (C). These are deterministic and reproduce exactly. The
solve times are the honest classical baseline, which no quantum result in this notebook competes with.
""")

code(module("solvers"))

code(r"""
OPTIMA, rows = {}, []
for name in INSTANCES:
    inst = build_instance(name)
    r, b = solve_exact(inst), solve_bruteforce(inst)
    assert abs(r["cost"] - b["cost"]) < 1e-6, "CP-SAT and exhaustive search disagree"
    OPTIMA[name] = r["cost"]
    rows.append({"Instance": name, "Binary variables": n_variables(inst),
                 "Optimum (EUR)": round(r["cost"], 2), "Facilities opened": r["opened"],
                 "CP-SAT time (ms)": round(r["wall_seconds"] * 1000, 1),
                 "Exhaustive search agrees": True})
pd.DataFrame(rows)
""")

code(module("maps"))

code(r"""
# What is being decided: instance B and its cheapest network.
inst_B = build_instance("B")
display(Image(figure_instance_map(inst_B, solve_exact(inst_B),
                                  os.path.join(OUTPUT_DIR, "figure2_instance_map_B.png"))))
""")

# ------------------------------------------------------------------ step 4
md(r"""
---
## Step 4 — Translating the problem into a QUBO

A quantum annealer accepts exactly one input format, a **QUBO** (Quadratic Unconstrained Binary Optimization).
*Binary*: every variable is on or off. *Quadratic*: costs attach to single variables or to pairs. *Unconstrained*:
**there is no way to state a rule that must hold.**

So the rules become fines. Breaking a rule is made expensive rather than forbidden:

| Rule | Penalty added to the cost |
|---|---|
| Each collection point assigned exactly once | `A × (number of assignments − 1)²` |
| No assignment to a closed facility | `B × x_ij × (1 − y_j)` |

**This creates the problem the study measures.** Nothing in the business problem says how large the fines should
be. Too low, and the solver breaks rules to save money, returning cheap but invalid answers. Too high, and the
fines swamp the cost signal, returning valid but expensive answers. `penalty_scale` multiplies both fines, and it
is **the only quantity the experiment varies**.
""")

code(module("qubo"))

# ------------------------------------------------------------------ step 5
md(r"""
---
## Step 5 — Verifying the translation before spending solver time

If the QUBO described a different problem, every later number would be meaningless. Classical simulated
annealing, which runs locally at no cost, solves the QUBO for instance A; reaching the exact optimum from Step 3
proves the translation is faithful.

The share of valid reads is already well below 100 % at this setting: the first sign of the effect under study.
""")

code(r"""
from dwave.samplers import SimulatedAnnealingSampler

inst_A = build_instance("A")
bqm, weights = build_bqm(inst_A, penalty_scale=0.25)
ss = SimulatedAnnealingSampler().sample(bqm, num_reads=1000, seed=42)
best, feasible_reads, total_reads = summarize(ss, inst_A)
print(f"QUBO terms:          {len(bqm.linear) + len(bqm.quadratic)}")
print(f"Classical optimum:   EUR {OPTIMA['A']:,.2f}")
print(f"QUBO best found:     EUR {best['cost']:,.2f}  (facilities {best['opened']})")
print(f"Valid reads:         {feasible_reads}/{total_reads}")
assert abs(best["cost"] - OPTIMA["A"]) < 1e-6, "MISMATCH: the QUBO is not the same problem"
print("\nMATCH: the QUBO and the original problem are the same problem.")
""")

# ------------------------------------------------------------------ step 6
md(r"""
---
## Step 6 — Connecting to D-Wave, and one test run

*Skipped in reproduce mode.*

D-Wave's quantum processors (QPUs, for *quantum processing unit*) are reached over the internet through the Leap
service. Listing them is free. The notebook then selects the processor by name, because the default can change between
sessions and which processor produced a result is a reportable fact; its `chip_id` is written on every row.

**Minor embedding.** The processor's qubits are not all connected to each other, so a variable that interacts
with many others is represented by a chain of physical qubits. `EmbeddingComposite` computes this mapping. A
problem therefore uses more qubits than it has variables, and chains can break if their qubits disagree; both
are recorded.

The single test run costs about one second of QPU time and confirms credentials, embedding and logging before
the full experiment. Its row is saved to `smoke_test.csv`, separately from the experimental data, so every
billed submission is accounted for.
""")

code(r"""
if not RUN_EXPERIMENT:
    print("Reproduce mode: no connection to D-Wave.")
else:
    from dwave.cloud import Client
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

    with Client.from_config() as client:
        solvers = client.get_solvers()
        qpus = [s.name for s in solvers if s.properties.get("category") == "qpu"]
        hybrids = [s.name for s in solvers if s.properties.get("category") == "hybrid"]
    print("QPUs available:   ", qpus)
    print("Hybrid available: ", hybrids)

    QPU_NAME = PREFERRED_QPU if PREFERRED_QPU in qpus else qpus[0]
    if QPU_NAME != PREFERRED_QPU:
        print(f"NOTE: {PREFERRED_QPU} is not available; using {QPU_NAME}. "
              "Results are not directly comparable with the archived run.")
    qpu = DWaveSampler(solver=QPU_NAME)
    qpu_sampler = EmbeddingComposite(qpu)
    CHIP_ID = qpu.properties.get("chip_id", QPU_NAME)
    hybrid_sampler = LeapHybridSampler()
    HYBRID_NAME = getattr(getattr(hybrid_sampler, "solver", None), "name", "LeapHybridSampler default")
    print(f"\nQPU:    {CHIP_ID} | {len(qpu.nodelist)} qubits | {qpu.properties['topology']['type']} topology")
    print(f"Hybrid: {HYBRID_NAME}")
""")

code(module("experiment"))

code(r"""
if not RUN_EXPERIMENT:
    print("Reproduce mode: no test run.")
else:
    row = run_one(build_instance("A"), OPTIMA["A"], "qpu", 0.25, 0,
                  qpu_sampler=qpu_sampler, chip_id=CHIP_ID)
    pd.DataFrame([row]).to_csv(os.path.join(OUTPUT_DIR, "smoke_test.csv"), index=False)
    print(f"Best cost:      EUR {row['best_cost']:,.0f}  (optimum {OPTIMA['A']:,.0f})")
    print(f"Optimality gap: {row['optimality_gap_pct']:.1f} %")
    print(f"Valid reads:    {row['feasible_fraction']:.0%}")
    print(f"Qubits used:    {row['physical_qubits']:.0f} for 27 variables, longest chain {row['max_chain_length']:.0f}")
    print(f"Chain breaks:   {row['chain_break_fraction']:.1%}")
    print(f"QPU time:       {row['qpu_access_time_us'] / 1000:.1f} ms")
""")

# ------------------------------------------------------------------ step 7
md(r"""
---
## Step 7 — The experiment

*In reproduce mode this step loads the archived results instead of running anything.*

**Design.** One factor varies: `penalty_scale`, at 0.1, 0.25, 0.5, 1.0, 2.0 and 4.0. Instance, solver, processor,
read count (1,000) and annealing time (20 µs) are constant within each stage. Each condition is repeated 20 times,
because annealing is probabilistic and one run cannot separate an effect from chance.

| Stage | Solver | Instances | Repetitions | Solver time |
|---|---|---|---|---|
| 1 | Simulated annealing (classical) | A, B, C | 20 | none, runs locally (5–10 min) |
| 2 | Advantage QPU, automatic chain strength | B | 20 | ≈ 15 s |
| 3 | Advantage QPU, **fixed** chain strength | B | 20 | ≈ 15 s |
| 4 | Leap hybrid solver | B | 3 (≈ 3 s minimum each; archive: 6) | ≈ 55 s |
| 5 | Advantage QPU, automatic chain strength | A, C | 20 | ≈ 30 s |

**Why Stage 3 exists.** `EmbeddingComposite` derives chain strength from the size of the QUBO coefficients, and
the penalty weight sets those coefficients. Stage 2 therefore changes two things at once. Stage 3 fixes chain
strength at the value D-Wave's own heuristic (`uniform_torque_compensation`) gives at penalty 1.0 and holds it
across all six settings, to test whether the effect belongs to the formulation or to the embedding.

Results are written to disk after every stage, so an interrupted session loses at most one stage.

**About the archived run.** Some stages were executed more than once during the session: Stage 5 three times for
instance A and twice for instance C, and Stage 4 twice. The archive therefore holds 60 and 40 repetitions per
setting for instances A and C on the quantum processor, and 6 for the hybrid solver. All repetitions are
independent runs and are kept. The Leap platform billed 899 QPU submissions against 840 rows here; the
difference is the test run and cells interrupted before their results were saved, which the incremental saving
in this version prevents.
""")

code(r"""
ARCHIVE_CANDIDATES = ["../data/results/bfd_results.csv", "data/results/bfd_results.csv", "bfd_results.csv"]
RESULTS_FILE = os.path.join(OUTPUT_DIR, "bfd_results.csv")
results = []

def save_progress():
    pd.concat(results, ignore_index=True).to_csv(RESULTS_FILE, index=False)

ARCHIVE_URL = ("https://raw.githubusercontent.com/ismail-lut/bfd-quantum-demonstration/"
               "main/data/results/bfd_results.csv")

if not RUN_EXPERIMENT:
    archive = next((p for p in ARCHIVE_CANDIDATES if os.path.exists(p)), None)
    if archive is None:                      # e.g. opened in Colab straight from GitHub
        try:
            import urllib.request
            urllib.request.urlretrieve(ARCHIVE_URL, "bfd_results.csv")
            archive = "bfd_results.csv"
            print(f"Downloaded archived results from {ARCHIVE_URL}")
        except Exception as e:
            print(f"Could not download archived results ({e}).")
    if archive is None and IN_COLAB:
        from google.colab import files
        print("Upload data/results/bfd_results.csv from the repository:")
        archive = next(iter(files.upload()))
    if archive is None:
        raise FileNotFoundError("Archived results not found; expected data/results/bfd_results.csv")
    print(f"Loaded archived results from {archive}")
else:
    print("Stage 1: simulated annealing, all instances (no solver time)")
    for name in INSTANCES:
        results.append(run_sweep(build_instance(name), OPTIMA[name], "sa", progress=True))
    save_progress()
""")

code(r"""
if RUN_EXPERIMENT:
    print("Stage 2: QPU, instance B")
    results.append(run_sweep(build_instance("B"), OPTIMA["B"], "qpu", progress=True,
                             qpu_sampler=qpu_sampler, chip_id=CHIP_ID))
    save_progress()

    print("Stage 3: QPU with fixed chain strength, instance B")
    from dwave.embedding.chain_strength import uniform_torque_compensation
    FIXED_CHAIN_STRENGTH = float(uniform_torque_compensation(build_bqm(build_instance("B"), 1.0)[0]))
    print(f"  fixed chain strength = {FIXED_CHAIN_STRENGTH:,.0f}  (archived run: 45,843)")
    results.append(run_sweep(build_instance("B"), OPTIMA["B"], "qpu_fixed_chain", progress=True,
                             chain_strength=FIXED_CHAIN_STRENGTH,
                             qpu_sampler=qpu_sampler, chip_id=CHIP_ID))
    save_progress()

    print("Stage 4: hybrid solver, instance B")
    results.append(run_sweep(build_instance("B"), OPTIMA["B"], "hybrid", replications=HYBRID_REPLICATIONS,
                             progress=True, hybrid_sampler=hybrid_sampler))
    save_progress()

    print("Stage 5: QPU, instances A and C")
    for name in ["A", "C"]:
        results.append(run_sweep(build_instance(name), OPTIMA[name], "qpu", progress=True,
                                 qpu_sampler=qpu_sampler, chip_id=CHIP_ID))
        save_progress()
""")


# ------------------------------------------------------------------ step 8
md(r"""
---
## Step 8 — Figures and tables

Everything below reads only the results table, so it uses no solver time and can be rerun freely. The analysis is
descriptive: with a handful of settings, the aim is to show the size and shape of the effect, not to test
hypotheses statistically.

**Cost and gap statistics are computed over feasible runs only.** A run that returned no valid solution has no
meaningful cost; an invalid configuration can even cost less than the optimum. The share of valid reads is
computed over all runs. (Averaging cost over all runs instead would give, for example, EUR 5,705 for the quantum
processor at penalty 0.1 on instance B: below the best possible cost of EUR 8,253, which no valid answer can be.)
""")

code(module("analysis"))

code(r"""
data = load_results(archive if not RUN_EXPERIMENT else RESULTS_FILE)
print(f"{len(data)} runs\n")
print(data.groupby(["solver", "instance"]).size().rename("runs").to_string())
print(f"\nQPU access time recorded: {data.qpu_access_time_us.sum() / 1e6:.1f} s")
print(f"Hybrid time charged:      {data.charge_time_us.sum() / 1e6:.1f} s")
""")

md(r"""
### Figure 1 — Solution quality against penalty weight, instance B

Shaded bands show one standard deviation across repetitions. A cross on the baseline marks a condition in which no
repetition found a valid solution; a hollow marker, one valid in only a minority of repetitions. Low penalties
fail by returning invalid answers; high penalties return valid answers of steadily worse cost.
""")

code(r"""
display(Image(figure_1(data, "B", os.path.join(OUTPUT_DIR, "figure1_penalty_sensitivity.png"))))
""")

md(r"""
### Table 1 — Instance B, by penalty setting and solver
""")

code(r"""
t1 = table_1(data, "B")
t1.to_csv(os.path.join(OUTPUT_DIR, "table1_instance_B.csv"), index=False)
t1
""")

md(r"""
### Table 2 — Robustness across instances

The spread column states the finding in one number: how far solution quality moves on an identical problem when
only the penalty weight changes.
""")

code(r"""
t2 = table_2(data)
t2.to_csv(os.path.join(OUTPUT_DIR, "table2_robustness.csv"), index=False)
t2
""")

md(r"""
### Table 3 — Minor-embedding overhead
""")

code(r"""
t3 = table_embedding(data)
t3.to_csv(os.path.join(OUTPUT_DIR, "table3_embedding.csv"), index=False)
t3
""")

md(r"""
### Chain breaks

Reported unrounded, because they are of order 0.01 % of chains per read: small enough to round to 0.0 % in
Table 3, but not zero.
""")

code(r"""
chain_break_summary(data)
""")

md(r"""
### Is the quantum processor more sensitive than the classical algorithm?

A quantum processor works with limited analog precision, so one might expect large penalties, which squeeze the
real cost differences into a small part of its range, to hurt it more than a classical algorithm solving the same
input. In the archived run they did not: the quantum processor is uniformly somewhat worse, with no widening
trend as the penalty grows.
""")

code(r"""
g = aggregate(data)
p = g[(g.instance == "B") & g.solver.isin(["sa", "qpu"])].pivot(
    index="penalty_scale", columns="solver", values="mean_gap")
p["qpu_minus_sa_pp"] = p["qpu"] - p["sa"]
p.round(1)
""")

md(r"""
---
## What the results show

1. **The whole process works.** A recycling-logistics problem was rewritten, checked against its known best answer
   and solved on a quantum processor, which found the best answer on the smallest instance.
2. **The penalty weight alone moved answer quality by 21.4 percentage points** on instance B on the quantum
   processor, with everything else held constant, and by 21.5 points for classical simulated annealing.
3. **The failures look very different.** Penalties too small give no valid answer, which is obvious. Penalties too
   large give valid answers 20 to 30 % more expensive than necessary, and nothing in the output says so.
4. **The hardware did not make the effect worse.** The quantum processor was 6 to 11 points behind simulated
   annealing at every setting, with no widening trend.
5. **On real hardware, penalties and chain strength are tied together.** Holding chain strength fixed reversed the
   direction of the effect and widened it to 50.3 points.
6. **Broken chains do not explain it.** They were rare (on average 0.012 % of chains per read, at most 0.33 %) and
   did not change with the setting, while answer quality changed by tens of points.
7. **The best setting does not carry over between instances**, and **the conventional exact solver was best at
   every size.**

**Limits.** One problem type, three small synthetic instances, one way of setting the penalties, one processor and
one session. The numbers show how large the effect can be; they do not estimate its size for real networks.
""")

# ------------------------------------------------------------------ write
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True}})
OUT.parent.mkdir(exist_ok=True)
nbformat.write(nb, OUT)
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    nb = nbformat.read(OUT, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbformat.write(nb, OUT)
    print("executed in reproduce mode and saved with outputs")
