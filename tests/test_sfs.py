import numpy as np
from sigconfide.estimates.sfs import sample_sfs

def test_sfs_algorithm_basic():
    """
    Test that the SFS algorithm appropriately maintains P * E = M
    and returns samples and errors.
    """
    
    # Construct a valid synthetic P and E
    # (K=3, N=2)
    P_orig = np.array([
        [0.5, 0.1],
        [0.3, 0.4],
        [0.2, 0.5]
    ])
    
    # (N=2, G=3)
    E_orig = np.array([
        [100, 50,  10],
        [ 20, 80, 150]
    ])
    
    M_orig = P_orig @ E_orig
    
    # Run the raw SFS
    np.random.seed(42) # For reproducibility
    exposures, frob_errors, errors = sample_sfs(m=M_orig, P=P_orig, E=E_orig, max_iter=1500, check=500, eps=1e-10)
    
    # Check shape
    n_samples = exposures.shape[-1]
    assert exposures.shape == (2, 3, n_samples)  # N=2, G=3, R=n_samples
    assert len(errors) == n_samples
    assert n_samples <= 1500
    assert n_samples > 0
    assert np.all(exposures >= -1e-10)
    
    # Check error is stable (should be close to 0 since P*E is theoretically constant in SFS)
    # The KL divergence between M and P@E should be quite small
    assert np.all(errors < 1e-4)
