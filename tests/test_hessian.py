import unittest
import numpy as np
from sigconfide.estimates.hessian import hessianSigExposures

class TestHessianSigExposures(unittest.TestCase):
    def test_hessian_sampling_basic(self):
        # Synthetic data setup
        # 3 Signatures, 96 mutation types (reduced to 5 for simplicity in this unit test if dimensions allow, 
        # but the code expects 96 to match P.shape[0]. Let's make a small P and m that match.)
        
        # Create a small valid P and m for testing logic, not biology
        # N=3 signatures, K=5 mutation types
        # Note: Code checks if len(m) == P.shape[0].
        
        np.random.seed(42)
        K = 10
        N = 3
        P = np.random.rand(K, N)
        # Normalize P columns
        P = P / P.sum(axis=0)
        
        # Ground truth exposures
        true_exp = np.array([0.5, 0.3, 0.2])
        m = np.dot(P, true_exp)
        # Add a little noise to m so it's not a perfect fit
        m_noisy = m + np.random.normal(0, 0.01, size=K)
        m_noisy = np.abs(m_noisy) # Ensure non-negative
        m_noisy = m_noisy / m_noisy.sum()
        
        R = 100
        mutation_count = 1000 # Dummy count
        
        def mock_solve(m, P):
            # Simple unconstrained least squares for testing: (P^T P)^-1 P^T m
            # Then project to simplex roughly
            res = np.linalg.lstsq(P, m, rcond=None)[0]
            res[res < 0] = 0
            res /= res.sum()
            return res
            
        exposures, errors = hessianSigExposures(m_noisy, P, R, mutation_count=mutation_count, decomposition_method=mock_solve)
        
        # Checks
        self.assertEqual(exposures.shape, (N, R))
        self.assertEqual(errors.shape, (R,))
        
        # Check that means are somewhat close to the optimum (within reason)
        # Calculate optimum using the same mock method
        x_hat = mock_solve(m_noisy, P)
        
        sample_mean = exposures.mean(axis=1)
        
        # Differences should be small
        np.testing.assert_allclose(sample_mean, x_hat, atol=0.05)
        
        # Check normalization
        np.testing.assert_allclose(exposures.sum(axis=0), 1.0, atol=1e-6)

    def test_hessian_sampling_shapes(self):
        # Test with strictly 96 channels if needed, but the function is generic on shapes
        # as long as m and P match.
        K = 96
        N = 5
        P = np.random.rand(K, N)
        P /= P.sum(axis=0)
        m = np.random.rand(K)
        m /= m.sum()
        
        R = 10
        mutation_count = 500
        
        # Mock solver
        def mock_solve(m, P):
            res = np.linalg.lstsq(P, m, rcond=None)[0]
            res[res < 0] = 0
            res /= res.sum()
            return res

        exposures, errors = hessianSigExposures(m, P, R, mutation_count, decomposition_method=mock_solve)
        self.assertEqual(exposures.shape, (N, R))
        self.assertEqual(errors.shape, (R,))

if __name__ == '__main__':
    unittest.main()
