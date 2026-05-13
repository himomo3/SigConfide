import numpy as np
from sigconfide.utils.utils import kl_divergence, FrobeniusNorm
def _amat(lambda_val, s, smix, N):
    """
    Creates the transformation matrix given lambda, two signature indices s and smix, and dimension N.
    """
    A = np.eye(N)
    A[s, s] = 1.0 - lambda_val
    A[smix, s] = lambda_val
    return A

def _lambda_range(P, E, s, smix):
    """
    Calculates the feasible interval [lmin, lmax] of lambda given matrices P and E,
    and the indices s and smix strings to mix.
    """
    # Exract relevant columns from P and rows from E
    p1 = P[:, s]
    p2 = P[:, smix]
    pdiff = p1 - p2
    
    # Calculate lmin
    neg_mask = pdiff < -1e-12
    lmin = 0.0
    if np.any(neg_mask):
        frac1 = p1[neg_mask] / pdiff[neg_mask]
        lmin = np.max(frac1)
        
    # Calculate lmax
    e1 = E[s, :]
    e2 = E[smix, :]
    non_zero_mask = (e1 > 1e-12) | (e2 > 1e-12)
    lmax = 1.0
    if np.any(non_zero_mask):
        frac2 = e2[non_zero_mask] / (e1[non_zero_mask] + e2[non_zero_mask])
        lmax = np.min(frac2)
        
    return lmin, lmax

def sample_sfs(m, P, E, max_iter=100000, check=1000, beta=0.5, eps=1e-10):
    """
    Finds the Set of Feasible Solutions (SFS) from a given solution of matrices P and E.
    
    This function iterativelty explores the valid non-negative transformations
    of an NMF solution without changing the product P * E.
    
    Parameters:
        m (numpy.ndarray): Observed tumor profile vector/matrix.
        P (numpy.ndarray): Signature matrix (K x N), e.g., 96 mutation types by N signatures.
        E (numpy.ndarray): Exposure matrix (N x G), e.g., N signatures by G patients.
        max_iter (int): Maximum number of iterations for the sampling algorithm.
        check (int): Number of iterations between checking for convergence.
        beta (float): Shape parameter in the beta distribution to sample lambda.
        eps (float): Epsilon for the stopping criteria based on average change.
        
    Returns:
        tuple: A tuple containing four numpy arrays.
            - exposures (numpy.ndarray): Matrix of signature exposures for each sample.
            - signatures (numpy.ndarray): Stacked signature matrices corresponding to the exposures.
            - frob_errors (numpy.ndarray): Estimation error (Frobenius norm).
            - errors (numpy.ndarray): Estimation error (KL divergence).
    """
    
    # Ensure working copies to prevent modifying strictly read-only inputs
    P_current = np.copy(P).astype(np.float64)
    E_current = np.copy(E).astype(np.float64)
    
    K, N = P_current.shape
    
    is_1d = False
    if E_current.ndim == 1:
        E_current = E_current.reshape(N, -1)
        is_1d = True
        
    _, G = E_current.shape
    
    m_current = np.copy(m).astype(np.float64)
    if m_current.ndim == 1:
        m_current = m_current.reshape(K, -1)
    
    all_E = []
    all_P = []
    all_errors = []
    all_frob_errors = []
    
    # Matrices to store histories during the checking window
    p_history = np.zeros((check + 1, K, N))
    diffnew = 1.0
    diffold = 0.0
    
    P_min = np.copy(P_current)
    P_max = np.copy(P_current)
    
    # Initial cleanup to avoid numeric issues around 0
    P_current[P_current < 1e-10] = 0
    E_current[E_current < 1e-10] = 0
    
    for i in range(max_iter):
        # We cycle through all columns (signatures) in one pass
        for s in range(N):
            smix = np.random.randint(0, N - 1)
            if smix >= s:
                smix += 1  # Ensures smix != s and drawn uniformly from available indices
                
            lmin, lmax = _lambda_range(P_current, E_current, s, smix)
            
            x = np.random.beta(beta, beta)
            lambda_val = lmin * x + lmax * (1 - x)
            
            if abs(lambda_val) > 1e-10 and (1.0 - lambda_val) != 0.0:
                # Update P
                p_s_new = P_current[:, s] * (1.0 - lambda_val) + P_current[:, smix] * lambda_val
                
                # Update E
                # Ainv has Ainv[s,s] = 1/(1-lambda), Ainv[smix,s] = -lambda/(1-lambda)
                # So Row s becomes 1/(1-lambda) * Row s
                # Row smix becomes Row smix + -lambda/(1-lambda) * Row s
                e_s_new = E_current[s, :] * (1.0 / (1.0 - lambda_val))
                e_smix_new = E_current[smix, :] + E_current[s, :] * (-lambda_val / (1.0 - lambda_val))
                
                P_current[:, s] = p_s_new
                E_current[s, :] = e_s_new
                E_current[smix, :] = e_smix_new
                
                # Clean small numeric errors
                P_current[:, s][P_current[:, s] < 1e-10] = 0
                E_current[s, :][E_current[s, :] < 1e-10] = 0
                E_current[smix, :][E_current[smix, :] < 1e-10] = 0
        
        m_approx = np.dot(P_current, E_current)
        err = kl_divergence(m_current, m_approx)
        frob_err = FrobeniusNorm(m_current, P_current, E_current)
        
        all_E.append(np.copy(E_current))
        all_P.append(np.copy(P_current))
        all_errors.append(err)
        all_frob_errors.append(frob_err)
        
        # Store state in history for this check period
        iter_mod = i % check
        p_history[iter_mod] = P_current
        
        # Check convergence metrics at the end of a check block
        if i > 0 and iter_mod == check - 1:
            p_history[-1] = P_min
            P_min_batch = np.min(p_history, axis=0)
            
            p_history[-1] = P_max
            P_max_batch = np.max(p_history, axis=0)
            
            prob_diff = np.ptp(p_history[:-1], axis=0)
            diffnew = np.mean(prob_diff)
            
            P_min = P_min_batch
            P_max = P_max_batch
            
            if (diffnew - diffold) < eps:
                break
            else:
                diffold = diffnew
                
    exposures = np.stack(all_E, axis=-1)
    signatures = np.stack(all_P, axis=-1)
    
    if is_1d or G == 1:
        # squeeze the middle dimension so it becomes (N, R)
        if exposures.ndim == 3 and exposures.shape[1] == 1:
            exposures = np.squeeze(exposures, axis=1)
            
    errors = np.array(all_errors)
    frob_errors = np.array(all_frob_errors)
            
    E_final = exposures[..., -1]
    E_whole = exposures
    P_final = signatures[..., -1]
    P_whole = signatures
    kl_errors_final = errors[-1]
    kl_errors_whole = errors
    frob_errors_final = frob_errors[-1]
    frob_errors_whole = frob_errors
            
    return E_final, E_whole, P_final, P_whole, kl_errors_final, kl_errors_whole, frob_errors_final, frob_errors_whole

def bootstrap_sfs(m, P, E, R=10, mutation_count=None, decomposition_method=None, max_iter=100000, check=1000, beta=0.5, eps=1e-10):
    """
    Combines bootstrap and SFS methods by generating bootstrap replicates of exposures
    and running SFS on each replicate.
    """
    from sigconfide.estimates.bootstrap import bootstrapSigExposures

    E_boot, _, _ = bootstrapSigExposures(m, P, R, mutation_count=mutation_count, decomposition_method=decomposition_method)

    all_E = []
    all_E_whole = []
    all_P = []
    all_P_whole = []
    all_kl = []
    all_kl_whole = []
    all_frob = []
    all_frob_whole = []

    for r in range(R):
        if E_boot.ndim == 3:
            E_r = E_boot[:, :, r]
        else:
            E_r = E_boot[:, r]

        E_r_out, E_whole_r, P_r, P_whole_r, kl_r, kl_whole_r, frob_r, frob_whole_r = sample_sfs(
            m, P, E_r, max_iter=max_iter, check=check, beta=beta, eps=eps
        )

        all_E.append(E_r_out)
        all_E_whole.append(E_whole_r)
        all_P.append(P_r)
        all_P_whole.append(P_whole_r)
        all_kl.append(kl_r)
        all_kl_whole.append(kl_whole_r)
        all_frob.append(frob_r)
        all_frob_whole.append(frob_whole_r)

    # Concatenate the results from all replicates
    final_E = np.stack(all_E, axis=-1)
    final_E_whole = np.concatenate(all_E_whole, axis=-1)
    final_P = np.stack(all_P, axis=-1)
    final_P_whole = np.concatenate(all_P_whole, axis=-1)
    final_kl = np.array(all_kl)
    final_kl_whole = np.concatenate(all_kl_whole, axis=-1)
    final_frob = np.array(all_frob)
    final_frob_whole = np.concatenate(all_frob_whole, axis=-1)

    return final_E, final_E_whole, final_P, final_P_whole, final_kl, final_kl_whole, final_frob, final_frob_whole

def bootstrap_poisson_sfs(m, P, E, R=10, mutation_count=None, decomposition_method=None, max_iter=100000, check=1000, beta=0.5, eps=1e-10):
    """
    Combines Poisson bootstrap and SFS methods by generating Poisson bootstrap replicates of exposures
    and running SFS on each replicate.
    """
    from sigconfide.estimates.bootstrap import bootstrapPoissonSigExposures

    E_boot, _, _ = bootstrapPoissonSigExposures(m, P, R, mutation_count=mutation_count, decomposition_method=decomposition_method)

    all_E = []
    all_E_whole = []
    all_P = []
    all_P_whole = []
    all_kl = []
    all_kl_whole = []
    all_frob = []
    all_frob_whole = []

    for r in range(R):
        if E_boot.ndim == 3:
            E_r = E_boot[:, :, r]
        else:
            E_r = E_boot[:, r]

        E_r_out, E_whole_r, P_r, P_whole_r, kl_r, kl_whole_r, frob_r, frob_whole_r = sample_sfs(
            m, P, E_r, max_iter=max_iter, check=check, beta=beta, eps=eps
        )

        all_E.append(E_r_out)
        all_E_whole.append(E_whole_r)
        all_P.append(P_r)
        all_P_whole.append(P_whole_r)
        all_kl.append(kl_r)
        all_kl_whole.append(kl_whole_r)
        all_frob.append(frob_r)
        all_frob_whole.append(frob_whole_r)

    # Concatenate the results from all replicates
    final_E = np.stack(all_E, axis=-1)
    final_E_whole = np.concatenate(all_E_whole, axis=-1)
    final_P = np.stack(all_P, axis=-1)
    final_P_whole = np.concatenate(all_P_whole, axis=-1)
    final_kl = np.array(all_kl)
    final_kl_whole = np.concatenate(all_kl_whole, axis=-1)
    final_frob = np.array(all_frob)
    final_frob_whole = np.concatenate(all_frob_whole, axis=-1)

    return final_E, final_E_whole, final_P, final_P_whole, final_kl, final_kl_whole, final_frob, final_frob_whole
