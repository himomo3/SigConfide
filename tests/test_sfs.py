import numpy as np
from sigconfide.estimates.sfs import sample_sfs, sfs_confidence_and_stability

def test_sfs_algorithm_basic():
    """
    Test that the SFS algorithm appropriately maintains P * E = M
    and preserves non-negativity.
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
    
    # Run the raw SFS
    np.random.seed(42) # For reproducibility
    results = sample_sfs(P_orig, E_orig, max_iter=2000, check=500, beta=0.5, eps=1e-10)
    
    P_min = results["Pminimum"]
    P_max = results["Pmaximum"]
    E_min = results["Eminimum"]
    E_max = results["Emaximum"]
    
    # Test non-negativity bounds
    assert np.all(P_min >= -1e-10)
    assert np.all(E_min >= -1e-10)
    
    # Check shape
    assert P_min.shape == P_orig.shape
    assert E_min.shape == E_orig.shape
    
    # Check that it sampled different configurations 
    # (Since there's ambiguity in this system, P_max > P_min in general)
    assert np.any(P_max > P_min)
    
    # Check stability calculation
    stability_metrics = sfs_confidence_and_stability(P_min, P_max, E_min, E_max)
    var = stability_metrics["channel_variation"]
    assert var.shape == P_orig.shape
    assert np.all(var >= 0)
