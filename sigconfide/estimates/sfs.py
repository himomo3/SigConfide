import numpy as np

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

def sample_sfs(P, E, max_iter=100000, check=1000, beta=0.5, eps=1e-10):
    """
    Finds the Set of Feasible Solutions (SFS) from a given solution of matrices P and E.
    
    This function iterativelty explores the valid non-negative transformations
    of an NMF solution without changing the product P * E.
    
    Parameters:
        P (numpy.ndarray): Signature matrix (K x N), e.g., 96 mutation types by N signatures.
        E (numpy.ndarray): Exposure matrix (N x G), e.g., N signatures by G patients.
        max_iter (int): Maximum number of iterations for the sampling algorithm.
        check (int): Number of iterations between checking for convergence.
        beta (float): Shape parameter in the beta distribution to sample lambda.
        eps (float): Epsilon for the stopping criteria based on average change.
        
    Returns:
        dict: A dictionary containing the SFS findings:
            - "avgChangeFinal": The final average change metric.
            - "totalIter": The number of iterations completed.
            - "Pminimum": Minimum sampled values for each entry in P.
            - "Pmaximum": Maximum sampled values for each entry in P.
            - "Eminimum": Minimum sampled values for each entry in E.
            - "Emaximum": Maximum sampled values for each entry in E.
    """
    
    # Ensure working copies to prevent modifying strictly read-only inputs
    P_current = np.copy(P).astype(np.float64)
    E_current = np.copy(E).astype(np.float64)
    
    K, N = P_current.shape
    _, G = E_current.shape
    
    # Track the extremes
    P_min = np.copy(P_current)
    P_max = np.copy(P_current)
    E_min = np.copy(E_current)
    E_max = np.copy(E_current)
    
    # Matrices to store histories during the checking window
    p_history = np.zeros((check + 1, K, N))
    e_history = np.zeros((check + 1, N, G))
    
    diffnew = 1.0
    diffold = 0.0
    
    # Initial cleanup to avoid numeric issues around 0
    P_current[P_current < 1e-10] = 0
    E_current[E_current < 1e-10] = 0
    
    total_iter = 0
    
    for i in range(max_iter):
        # We cycle through all columns (signatures) in one pass
        for s in range(N):
            smix = np.random.randint(0, N - 1)
            if smix >= s:
                smix += 1  # Ensures smix != s and drawn uniformly from available indices
                
            lmin, lmax = _lambda_range(P_current, E_current, s, smix)
            
            # Sample mixing proportion from Beta distribution
            # Instead of raw gamma draws like in cpp, we can just draw from Beta directly
            # The cpp did: gvar = randg(2, beta, 1.0); x = gvar[0]/sum(gvar); which is exactly Beta(beta, beta)
            x = np.random.beta(beta, beta)
            
            lambda_val = lmin * x + lmax * (1 - x)
            
            if abs(lambda_val) > 1e-10 and (1.0 - lambda_val) != 0.0:
                # Update P
                # A[s,s] = 1-lambda, A[smix,s] = lambda
                # P = P * A
                # This only affects column s of P
                p_s_new = P_current[:, s] * (1.0 - lambda_val) + P_current[:, smix] * lambda_val
                
                # Update E
                # E = A^-1 * E
                # A^-1 has (s,s) = 1/(1-lambda) and (smix,s) = -lambda/(1-lambda)
                # This only affects row s of E
                e_s_new = E_current[s, :] * (1.0 / (1.0 - lambda_val)) + E_current[smix, :] * (-lambda_val / (1.0 - lambda_val))
                
                P_current[:, s] = p_s_new
                E_current[s, :] = e_s_new
                
                # Clean small numeric errors
                P_current[:, s][P_current[:, s] < 1e-10] = 0
                E_current[s, :][E_current[s, :] < 1e-10] = 0
        
        # Store state in history for this check period
        iter_mod = i % check
        p_history[iter_mod] = P_current
        e_history[iter_mod] = E_current
        
        # Check convergence metrics at the end of a check block
        if i > 0 and iter_mod == check - 1:
            # Add previous check period's min/max as the first element of history to compare against
            p_history[-1] = P_min
            e_history[-1] = E_min
            
            # Update global min/max across this batch
            P_min_batch = np.min(p_history, axis=0)
            P_max_batch = np.max(p_history, axis=0)
            
            # Replace the -1 index with current maxes to compute global rolling max
            p_history[-1] = P_max
            P_max_batch = np.max(p_history, axis=0)
            
            E_min_batch = np.min(e_history, axis=0)
            E_max_batch = np.max(e_history, axis=0)
            
            # Replace the -1 index with current maxes to compute global rolling max
            e_history[-1] = E_max
            E_max_batch = np.max(e_history, axis=0)
            
            # Compute range (max - min) for each entry in P across this batch to find variation
            # We want the max variation across the history we just captured to measure if things are still expanding
            prob_diff = np.ptp(p_history[:-1], axis=0)
            diffnew = np.mean(prob_diff)
            
            # Commit new global min/max bounds
            P_min = P_min_batch
            P_max = P_max_batch
            E_min = E_min_batch
            E_max = E_max_batch
            
            if (diffnew - diffold) < eps:
                total_iter = i + 1
                break
            else:
                diffold = diffnew
                
        total_iter = i + 1

    return {
        "avgChangeFinal": diffnew,
        "totalIter": total_iter,
        "Pminimum": P_min,
        "Pmaximum": P_max,
        "Eminimum": E_min,
        "Emaximum": E_max
    }

def sfs_confidence_and_stability(P_min, P_max, E_min, E_max):
    """
    Analyzes the output of sample_sfs to determine signature stability and exposure confidence.
    
    Parameters:
        P_min (numpy.ndarray): Minimum signature profiles (K x N).
        P_max (numpy.ndarray): Maximum signature profiles (K x N).
        E_min (numpy.ndarray): Minimum exposures (N x G).
        E_max (numpy.ndarray): Maximum exposures (N x G).
        
    Returns:
        dict:
            - "signature_variation": The normalized variation for each entry in P to identify unstable signatures.
            - "exposure_confidence_bounds": (min, max) range of exposure for each signature in each sample.
    """
    K, N = P_min.shape
    
    # Calculate geometric mean or arithmetic mean as a denominator
    # For numeric stability, add small epsilon
    P_mean = (P_max + P_min) / 2.0 + 1e-12
    
    # Variation in P (max spread normalized by mean presence)
    variation = (P_max - P_min) / P_mean
    
    # We could aggregate this variation per signature by averaging over mutation types
    # Higher values indicate more instability due to non-uniqueness
    signature_variability = np.mean(variation, axis=0)
    
    # For exposures, we provide the raw intervals 
    # E_min indicates the lower bound. If a signature's e_min > 0, we're confident it's present.
    return {
        "signature_instability_score": signature_variability,  # Array of length N
        "channel_variation": variation,  # K x N
        "E_bounds": (E_min, E_max)       # Tuple of arrays N x G
    }
