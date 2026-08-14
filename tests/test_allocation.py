import numpy as np
from logit.models.allocation import DependencyGraphClustering, CapitalAllocationOptimizer

def test_dependency_graph_affinity():
    clustering = DependencyGraphClustering()
    branching = np.array([[0.1, 0.2], [0.2, 0.1]])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    semantic = np.array([[1.0, 0.8], [0.8, 1.0]])
    affinity = clustering.build_affinity_matrix(branching, corr, semantic)
    laplacian = clustering.compute_normalized_laplacian(affinity)
    embedding = clustering.extract_spectral_embedding(laplacian, n_clusters=1)
    assert affinity.shape == (2, 2)
    assert laplacian.shape == (2, 2)
    assert embedding.shape == (2, 1)

def test_capital_allocation_lp():
    optimizer = CapitalAllocationOptimizer()
    returns = np.array([0.1, 0.15])
    caps = np.array([10000.0, 10000.0])
    allocation = optimizer.solve_allocation(
        expected_returns=returns,
        borrower_caps=caps,
        pool_capacity=15000.0,
        market_exposures=np.array([[1.0, 0.0], [0.0, 1.0]]),
        market_caps=np.array([20000.0, 20000.0]),
        cluster_limits=np.array([20000.0, 20000.0]),
        cluster_membership=np.array([0, 1]),
        treasury_equity=5000.0,
        expected_losses=np.array([100.0, 100.0]),
        hawkes_multiplier=1.2
    )
    assert len(allocation) == 2
    assert np.sum(allocation) <= 15000.0 + 1e-5
