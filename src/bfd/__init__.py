"""Bounded feasibility demonstration: penalty-weight sensitivity of a
reverse logistics QUBO on D-Wave quantum annealing hardware."""

from .instances import INSTANCES, SEED, build_instance, decode, n_variables
from .qubo import build_bqm, build_qubo
from .solvers import solve_bruteforce, solve_exact, summarize

__version__ = "1.0.0"
