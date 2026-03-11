import numpy as np
from sigconfide.estimates.bootstrap import bootstrapSigExposures

def mock_decomp(m, P):
    # Just returns uniform exposures
    return np.ones(P.shape[1]) / P.shape[1]

def test_bootstrap_kl_error():
    np.random.seed(42)
    P = np.array([
        [0.5, 0.1],
        [0.3, 0.4],
        [0.2, 0.5]
    ])
    m_counts = np.array([50, 30, 20])
    m_probs = m_counts / 100
    
    # Run bootstrap
    exps, frob_errs, kl_errs = bootstrapSigExposures(m=m_counts, P=P, R=10, decomposition_method=mock_decomp)
    
    assert exps.shape == (2, 10)
    assert len(frob_errs) == 10
    assert len(kl_errs) == 10
    assert np.all(kl_errs >= 0)
