"""
Utilities: Linear algebra helpers, numerical differentiation,
and structured logging.
"""
from qgt.utils.linalg import sparse_kron_chain, matrix_function_safe
from qgt.utils.finite_diff import central_difference, gradient_2d
from qgt.utils.logging import get_logger
