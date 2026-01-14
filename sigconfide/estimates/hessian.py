import numpy as np
from sigconfide.utils.utils import is_wholenumber, FrobeniusNorm

# Try to import decomposeQP, but don't fail if it's missing (allows testing without quadprog)
try:
    from sigconfide.decompose.qp import decomposeQP
except ImportError:
    decomposeQP = None

def hessianSigExposures(m, P, R, mutation_count=None, decomposition_method=None):
    """
    Obtain the uncertainty of signature exposures using Hessian-based sampling.

    This function calculates the optimal solution once and then samples from an
    approximate posterior distribution derived from the Hessian of the objective function.

    Parameters:
        m (numpy.ndarray): Observed tumor profile vector for a patient/sample.
            It should have a shape of (96, 1) and can represent mutation counts or mutation probabilities.
        P (numpy.ndarray): Signature profile matrix with a shape of (96, N),
            where N is the number of signatures.
        R (int): The number of samples to generate.
        mutation_count (int, optional): If 'm' is a vector of counts, then 'mutation_count' equals
            the summation of all the counts. If 'm' is probabilities, 'mutation_count' must be specified.
            Used for variance estimation.
        decomposition_method (function, optional): The method selected to get the optimal solution.
            Default is 'decomposeQP'.

    Returns:
        tuple: A tuple containing two numpy arrays.
            - exposures (numpy.ndarray): Matrix of signature exposures for each sample (column).
            - errors (numpy.ndarray): Estimation error for each sample (Frobenius norm).

    Raises:
        ValueError: If inputs are invalid (shape mismatch, missing counts).
    """

    if len(m) != P.shape[0]:
        raise ValueError("Length of vector 'm' and number of rows of matrix 'P' must be the same.")
    if m.shape[0] != P.shape[0]:
        raise ValueError("Elements of vector 'm' and rows of matrix 'P' must have the same names.")
    if P.shape[1] < 2:
        raise ValueError("Matrix 'P' must have at least 2 columns (signatures).")

    # If 'mutation_count' is not specified, 'm' has to contain counts
    if mutation_count is None:
        if all(is_wholenumber(val) for val in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError("Please specify 'mutation_count' or provide counts in 'm'.")

    # Handle default decomposition method
    if decomposition_method is None:
        if decomposeQP is None:
             raise ImportError("The default decomposition method 'decomposeQP' requires 'quadprog' which is not installed. Please install 'quadprog' or provide a custom 'decomposition_method'.")
        decomposition_method = decomposeQP

    # Normalize m to be a vector of probabilities
    m_normalized = m / np.sum(m)

    # 1. Calculate the optimal exposure (Maximum Likelihood Estimate equivalent for Least Squares)
    x_hat = decomposition_method(m_normalized, P)

    # 2. Calculate the Hessian Matrix H = P^T * P
    # The Hessian of the RSS objective function f(x) = ||Ax - b||^2 is 2 * A^T A.
    # We drop the factor of 2 as it scales both signal and noise or is absorbed in variance estimation,
    # but strictly for covariance estimation on probabilistic interpretation:
    # Cov = inverse(Fisher Information). For Gaussian noise, FI approx H.
    H = np.dot(P.T, P)

    # 3. Estimate Error Variance (sigma_squared)
    # sigma^2 = RSS / (N_observations - N_parameters)
    # Here N_observations = 96 (channels), N_params = number of signatures
    reconstructed = np.dot(P, x_hat)
    residuals = m_normalized - reconstructed
    rss = np.sum(residuals**2)
    
    n_obs = P.shape[0]
    n_params = P.shape[1]
    
    if n_obs > n_params:
        sigma_squared = rss / (n_obs - n_params)
    else:
        # Fallback if degrees of freedom are low (unlikely for 96 channels and <30 sigs)
        sigma_squared = rss / n_obs

    # Scale sigma by mutation count if we want to model Poisson-like variance structure?
    # Standard bootstrap simulates multinomial sampling.
    # Hessian method typically assumes Gaussian approximation.
    # The covariance of the estimate is Cov(x) = sigma^2 * inverse(H)
    
    # In approximate Bayesian terms or asymptotic normality of MLE:
    # Cov = inverse(Target Function Hessian)
    # If we treat the noise variance as derived from the fit:
    try:
        covariance_matrix = sigma_squared * np.linalg.inv(H)
    except np.linalg.LinAlgError:
         # Use pseudo-inverse if H is singular (collinear signatures)
        covariance_matrix = sigma_squared * np.linalg.pinv(H)

    # 4. Sample from Multivariate Normal Distribution
    # shape: (R, N_params)
    samples = np.random.multivariate_normal(mean=x_hat, cov=covariance_matrix, size=R)
    
    # Transpose to (N_params, R) for consistency with bootstrap output
    exposures = samples.T

    # 5. Post-processing
    # Project negative values to 0 (non-negative constraint approximation)
    exposures[exposures < 0] = 0
    
    # Renormalize to sum to 1 (simplex constraint)
    # Avoid division by zero
    col_sums = np.sum(exposures, axis=0)
    # If a column sum is 0 (unlikely but possible if all sampled negative), set uniformity or keep 0?
    # For safety, add epsilon or handle.
    valid_cols = col_sums > 0
    exposures[:, valid_cols] /= col_sums[valid_cols]
    
    # If any column summed to 0, it means the sample was far off into negative space. 
    # Fallback to x_hat for those pathological cases is safer than NaNs.
    if not np.all(valid_cols):
        exposures[:, ~valid_cols] = x_hat[:, np.newaxis]

    # 6. Compute estimation errors
    # G x R
    errors = np.vectorize(lambda i: FrobeniusNorm(m_normalized, P, exposures[:, i]))(range(exposures.shape[1]))

    return exposures, errors
