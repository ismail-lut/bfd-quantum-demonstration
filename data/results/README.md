# Raw results — hardware run of 5 November 2026

`bfd_results.csv` holds every recorded solver run from the experiment described in the main README: one row per run,
1,236 rows. It is kept exactly as produced. The analysis code never changes it; it only summarizes it.

## What is in it

| Solver label | Meaning | Instance A | Instance B | Instance C |
|---|---|---|---|---|
| `sa` | Simulated annealing (classical) | 120 | 120 | 120 |
| `qpu` | D-Wave quantum processor, automatic chain strength | 360 | 120 | 240 |
| `qpu_fixed_chain` | D-Wave quantum processor, chain strength fixed at 45,843 | | 120 | |
| `hybrid` | D-Wave Leap hybrid solver | | 36 | |

Each count is 6 penalty settings times the number of repetitions. Instance B, the main test case, has 20
repetitions per setting on the quantum processor and in simulated annealing, and 6 for the hybrid solver. Some
stages were carried out more than once during the session, which is why instances A and C have 60 and 40
quantum repetitions per setting and the hybrid solver 6 rather than the planned 3. Every repetition is an independent run and all are kept.

## Columns

| Column | Unit | Meaning |
|---|---|---|
| `instance` | | A, B or C (see `data/instances/`) |
| `n_variables` | | Number of yes/no decisions in the problem |
| `solver` | | Solver label, as above |
| `penalty_scale` | | The setting being varied: a multiplier on both penalty weights |
| `penalty_A`, `penalty_B` | EUR | The resulting fines for assigning a point to other than exactly one facility (`A`) and to a closed facility (`B`) |
| `replication` | | Repetition number within one setting; restarts at 0 each time a stage was run |
| `chip_id` | | Quantum processor that ran the job (quantum rows only) |
| `num_reads` | | Candidate answers requested per run |
| `annealing_time_us` | µs | Annealing time per read (quantum rows only) |
| `chain_strength` | | Chain strength used (quantum rows only) |
| `chain_break_fraction` | fraction | Share of chains broken per read, averaged over the run's reads |
| `max_chain_length` | qubits | Longest chain used to represent one decision |
| `physical_qubits` | qubits | Physical qubits used in total |
| `qpu_access_time_us` | µs | Quantum processor time reported by D-Wave |
| `charge_time_us` | µs | Solver time charged (hybrid rows) |
| `wall_seconds` | s | Elapsed time including network and queueing; for context only |
| `best_cost` | EUR | Cost of the best valid answer; if none was valid, the cost of the lowest-scoring answer, which **is not a valid solution** |
| `feasible` | true/false | Whether any answer in the run obeyed every rule |
| `optimality_gap_pct` | % | How far `best_cost` is above the lowest possible cost; empty when no valid answer was found |
| `feasible_fraction` | fraction | Share of the run's answers that obeyed every rule |

## Two cautions

1. **Average costs over valid rows only** (`feasible` is true). On other rows `best_cost` is the cost of an
   invalid configuration and can be below the lowest possible cost.
2. **Do not round chain breaks before checking whether they occurred.** Values are of order 0.0001 and print as
   0.0 % at one decimal place, but 667 of the 840 quantum rows are above zero. The largest is 0.33 %.

## Solver time as recorded by D-Wave

`leap_usage_2026-11-05.csv` records the usage shown on the D-Wave Leap dashboard after the run. It is the
authoritative record of solver time used.

| | In this file | Recorded by D-Wave | Difference |
|---|---|---|---|
| Quantum processor runs | 840 | 899 | 59 |
| Quantum processor time | 106.1 s | 114.481 s | 8.4 s |
| Hybrid solver runs | 36 | 36 | 0 |
| Hybrid solver time | 107.686 s | 107.686 s | 0 |

The 59 quantum runs recorded by D-Wave but absent here are a single connection test and runs from notebook cells
that were interrupted before their results were saved. `notebooks/archive/BFD_run_2026-11-05.ipynb` is the notebook
as executed for this run, kept as a record. The current notebook saves results after every
stage and stores the connection test separately, so the two records match in a new run.
