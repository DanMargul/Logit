import numpy as np
from logit.models.optimal_transport import SinkhornSolver, WassersteinRobustness

def test_sinkhorn_solver_convergence():
    p_emp = np.array([0.5, 0.5])
    w_target = np.array([0.5, 0.5])
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    solver = SinkhornSolver(epsilon=0.1, max_iter=100, tolerance=1e-6)
    plan = solver.solve(p_emp, w_target, cost_matrix)
    assert plan.shape == (2, 2)
    np.testing.assert_allclose(np.sum(plan, axis=1), p_emp, rtol=1e-4)
    np.testing.assert_allclose(np.sum(plan, axis=0), w_target, rtol=1e-4)

def test_wasserstein_robustness_addon():
    robustness = WassersteinRobustness(confidence_level=0.01)
    sizes = np.array([10000.0, 20000.0])
    discounts = np.array([0.1, 0.2])
    k = robustness.compute_structure_aware_lipschitz(sizes, discounts)
    addon = robustness.calculate_capital_addon(k, 0.15)
    assert k > 0.0
    assert addon > 0.0
