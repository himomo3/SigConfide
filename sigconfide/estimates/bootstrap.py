import numpy as np
import multiprocessing as mp
from sigconfide.utils.utils import is_wholenumber, FrobeniusNorm, kl_divergence

# Try to import decomposeQP, but don't fail if it's missing (allows testing without quadprog)
try:
    from sigconfide.decompose.qp import decomposeQP
except ImportError:
    decomposeQP = None

def _run_bootstrap_replicate(args):
    m_sample, P, decomposition_method = args
    return decomposition_method(m_sample, P)

def bootstrapSigExposures(m, P, R, mutation_count=None, decomposition_method=None, n_jobs=-2):
    """
    Obtain the bootstrap distribution of signature exposures for a tumor sample.

    This function allows obtaining the bootstrap distribution of the signature exposures for
    a specific tumor sample using a specified decomposition method.

    Parameters:
        m (numpy.ndarray): Observed tumor profile vector for a patient/sample.
            It should have a shape of (96, 1) and can represent mutation counts or mutation probabilities.
        P (numpy.ndarray): Signature profile matrix with a shape of (96, N),
            where N is the number of signatures (e.g., COSMIC: N=30).
        R (int): The number of bootstrap replicates.
        mutation_count (int, optional): If 'm' is a vector of counts, then 'mutation_count' equals
            the summation of all the counts. If 'm' is probabilities, 'mutation_count' must be specified.
        decomposition_method (function, optional): The method selected to get the optimal solution.
            It should be a function. Default is 'decomposeQP'.

    Returns:
        tuple: A tuple containing three numpy arrays.
            - exposures (numpy.ndarray): Matrix of signature exposures for each bootstrap replicate (column).
            - errors (numpy.ndarray): Estimation error for each bootstrap replicate (Frobenius norm).
            - kl_errors (numpy.ndarray): Estimation error for each bootstrap replicate (KL divergence).

    Raises:
        ValueError: If the length of vector 'm' and the number of rows of matrix 'P' do not match,
            if 'P' has less than 2 columns, if 'mutation_count' is not specified and 'm' does not contain counts.

    Examples:
        bootstrapSigExposures(tumorBRCA[:, 1], signaturesCOSMIC, 100, 2000, decomposeQP)
        sigsBRCA = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
        bootstrapSigExposures(tumorBRCA[:, 1], signaturesCOSMIC[:, sigsBRCA], 10, 1000, decomposeQP)
    """

    if decomposition_method is None:
        if decomposeQP is None:
             raise ImportError("The default decomposition method 'decomposeQP' requires 'quadprog' which is not installed. Please install 'quadprog' or provide a custom 'decomposition_method'.")
        decomposition_method = decomposeQP

    is_2d = m.ndim == 2 and m.shape[1] > 1
    if m.shape[0] != P.shape[0]:
        raise ValueError("Rows of matrix 'm' and rows of matrix 'P' must be the same.")

    K = m.shape[0]  # number of mutation types
    
    if n_jobs == -1:
        processes = min(R, mp.cpu_count())
    elif n_jobs == -2:
        processes = min(R, max(1, mp.cpu_count() // 2))
    else:
        processes = min(R, max(1, n_jobs))

    with mp.Pool(processes=processes) as pool:
        if is_2d:
            G = m.shape[1]
            if mutation_count is None:
                mutation_count = []
                for g in range(G):
                    if all(is_wholenumber(val) for val in m[:, g]):
                        mutation_count.append(int(m[:, g].sum()))
                    else:
                        raise ValueError("Please specify the parameter 'mutation_count' or provide mutation counts in parameter 'm'.")
            else:
                if isinstance(mutation_count, (int, float, np.integer)):
                    mutation_count = [int(mutation_count)] * G
    
            m = m / m.sum(axis=0)
            
            def bootstrap_sample_col(m_col, mc, K):
                mutations_sampled = np.random.choice(K, size=mc, p=m_col)
                return np.bincount(mutations_sampled, minlength=K) / mc
                
            exposures_all = []
            errors_all = []
            kl_errors_all = []
            
            for g in range(G):
                m_samples = [bootstrap_sample_col(m[:, g], mutation_count[g], K) for _ in range(R)]
                tasks = [(samp, P, decomposition_method) for samp in m_samples]
                exposures_g = np.column_stack(pool.map(_run_bootstrap_replicate, tasks))
                
                # normalize sum
                expos_sum = np.sum(exposures_g, axis=0)
                expos_sum[expos_sum == 0] = 1.0 # prevent division by zero
                exposures_g = exposures_g / expos_sum
                
                errors_g = np.vectorize(lambda i: FrobeniusNorm(m[:, g], P, exposures_g[:, i]))(range(exposures_g.shape[1]))
                
                m_approx = P @ exposures_g
                eps = 1e-10
                m_expanded = m[:, g][:, np.newaxis]
                kl_matrix = m_expanded * np.log((m_expanded + eps) / (m_approx + eps)) - m_expanded + m_approx
                kl_errors_g = np.sum(kl_matrix, axis=0)
                
                exposures_all.append(exposures_g)
                errors_all.append(errors_g)
                kl_errors_all.append(kl_errors_g)
                
            exposures = np.stack(exposures_all, axis=1) # (N, G, R)
            errors = np.stack(errors_all, axis=0)       # (G, R)
            kl_errors = np.stack(kl_errors_all, axis=0) # (G, R)
            
        else:
            m = m.flatten()
            if mutation_count is None:
                if all(is_wholenumber(val) for val in m):
                    mutation_count = int(m.sum())
                else:
                    raise ValueError("Please specify the parameter 'mutation_count' or provide mutation counts in parameter 'm'.")
    
            m = m / np.sum(m)
    
            def bootstrap_sample(m, mutation_count, K):
                mutations_sampled = np.random.choice(K, size=mutation_count, p=m)
                return np.bincount(mutations_sampled, minlength=K) / mutation_count
    
            m_samples = [bootstrap_sample(m, mutation_count, K) for _ in range(R)]
            tasks = [(samp, P, decomposition_method) for samp in m_samples]
            exposures = np.column_stack(pool.map(_run_bootstrap_replicate, tasks))
            
            exposures = exposures / np.sum(exposures, axis=0)
    
            errors = np.vectorize(lambda i: FrobeniusNorm(m, P, exposures[:, i]))(range(exposures.shape[1]))
            
            m_approx = P @ exposures
            eps = 1e-10
            m_expanded = m[:, np.newaxis]
            kl_matrix = m_expanded * np.log((m_expanded + eps) / (m_approx + eps)) - m_expanded + m_approx
            kl_errors = np.sum(kl_matrix, axis=0)

    return exposures, errors, kl_errors

def bootstrapPoissonSigExposures(m, P, R, mutation_count=None, decomposition_method=None, n_jobs=-2):
    """
    Obtain the Poisson bootstrap distribution of signature exposures for a tumor sample.

    This function allows obtaining the bootstrap distribution of the signature exposures for
    a specific tumor sample using a Poisson bootstrap approach and a specified decomposition method.

    Parameters:
        m (numpy.ndarray): Observed tumor profile vector for a patient/sample.
            It should have a shape of (96, 1) and can represent mutation counts or mutation probabilities.
        P (numpy.ndarray): Signature profile matrix with a shape of (96, N),
            where N is the number of signatures (e.g., COSMIC: N=30).
        R (int): The number of bootstrap replicates.
        mutation_count (int, optional): If 'm' is a vector of counts, then 'mutation_count' equals
            the summation of all the counts. If 'm' is probabilities, 'mutation_count' must be specified.
        decomposition_method (function, optional): The method selected to get the optimal solution.
            It should be a function. Default is 'decomposeQP'.

    Returns:
        tuple: A tuple containing three numpy arrays.
            - exposures (numpy.ndarray): Matrix of signature exposures for each bootstrap replicate (column).
            - errors (numpy.ndarray): Estimation error for each bootstrap replicate (Frobenius norm).
            - kl_errors (numpy.ndarray): Estimation error for each bootstrap replicate (KL divergence).
    """

    if decomposition_method is None:
        if decomposeQP is None:
             raise ImportError("The default decomposition method 'decomposeQP' requires 'quadprog' which is not installed. Please install 'quadprog' or provide a custom 'decomposition_method'.")
        decomposition_method = decomposeQP

    is_2d = m.ndim == 2 and m.shape[1] > 1
    if m.shape[0] != P.shape[0]:
        raise ValueError("Rows of matrix 'm' and rows of matrix 'P' must be the same.")

    K = m.shape[0]  # number of mutation types
    
    if n_jobs == -1:
        processes = min(R, mp.cpu_count())
    elif n_jobs == -2:
        processes = min(R, max(1, mp.cpu_count() // 2))
    else:
        processes = min(R, max(1, n_jobs))

    with mp.Pool(processes=processes) as pool:
        if is_2d:
            G = m.shape[1]
            if mutation_count is None:
                mutation_count = []
                for g in range(G):
                    if all(is_wholenumber(val) for val in m[:, g]):
                        mutation_count.append(int(m[:, g].sum()))
                    else:
                        raise ValueError("Please specify the parameter 'mutation_count' or provide mutation counts in parameter 'm'.")
            else:
                if isinstance(mutation_count, (int, float, np.integer)):
                    mutation_count = [int(mutation_count)] * G
    
            m = m / m.sum(axis=0)
            
            def bootstrap_poisson_sample_col(m_col, mc, K):
                sampled_counts = np.random.poisson(m_col * mc)
                sampled_sum = sampled_counts.sum()
                if sampled_sum == 0:
                    sampled_sum = 1.0 # prevent division by zero
                return sampled_counts / sampled_sum
                
            exposures_all = []
            errors_all = []
            kl_errors_all = []
            
            for g in range(G):
                m_samples = [bootstrap_poisson_sample_col(m[:, g], mutation_count[g], K) for _ in range(R)]
                tasks = [(samp, P, decomposition_method) for samp in m_samples]
                exposures_g = np.column_stack(pool.map(_run_bootstrap_replicate, tasks))
                
                # normalize sum
                expos_sum = np.sum(exposures_g, axis=0)
                expos_sum[expos_sum == 0] = 1.0 # prevent division by zero
                exposures_g = exposures_g / expos_sum
                
                errors_g = np.vectorize(lambda i: FrobeniusNorm(m[:, g], P, exposures_g[:, i]))(range(exposures_g.shape[1]))
                
                m_approx = P @ exposures_g
                eps = 1e-10
                m_expanded = m[:, g][:, np.newaxis]
                kl_matrix = m_expanded * np.log((m_expanded + eps) / (m_approx + eps)) - m_expanded + m_approx
                kl_errors_g = np.sum(kl_matrix, axis=0)
                
                exposures_all.append(exposures_g)
                errors_all.append(errors_g)
                kl_errors_all.append(kl_errors_g)
                
            exposures = np.stack(exposures_all, axis=1) # (N, G, R)
            errors = np.stack(errors_all, axis=0)       # (G, R)
            kl_errors = np.stack(kl_errors_all, axis=0) # (G, R)
            
        else:
            m = m.flatten()
            if mutation_count is None:
                if all(is_wholenumber(val) for val in m):
                    mutation_count = int(m.sum())
                else:
                    raise ValueError("Please specify the parameter 'mutation_count' or provide mutation counts in parameter 'm'.")
    
            m = m / np.sum(m)
    
            def bootstrap_poisson_sample(m, mutation_count, K):
                sampled_counts = np.random.poisson(m * mutation_count)
                sampled_sum = sampled_counts.sum()
                if sampled_sum == 0:
                    sampled_sum = 1.0
                return sampled_counts / sampled_sum
    
            m_samples = [bootstrap_poisson_sample(m, mutation_count, K) for _ in range(R)]
            tasks = [(samp, P, decomposition_method) for samp in m_samples]
            exposures = np.column_stack(pool.map(_run_bootstrap_replicate, tasks))
            
            exposures = exposures / np.sum(exposures, axis=0)
    
            errors = np.vectorize(lambda i: FrobeniusNorm(m, P, exposures[:, i]))(range(exposures.shape[1]))
            
            m_approx = P @ exposures
            eps = 1e-10
            m_expanded = m[:, np.newaxis]
            kl_matrix = m_expanded * np.log((m_expanded + eps) / (m_approx + eps)) - m_expanded + m_approx
            kl_errors = np.sum(kl_matrix, axis=0)

    return exposures, errors, kl_errors
