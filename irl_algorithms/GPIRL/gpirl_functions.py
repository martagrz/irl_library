import numpy as np


def invert_kernel(K, y):
    # Safe inversion of kernel matrix that uses SVD decomposition if Cholesky fails.
    # First, try Cholesky.
    try:
        L = np.linalg.cholesky(K)
        #assert K == np.dot(L, L.T.conj())
        a = np.eye(y.shape[0])
        u = np.linalg.solve(L.T.conj(), a)
        K_inverse = np.linalg.solve(L, u)
        #assert np.dot(K, K_inverse) == np.eye(y)
    except:
        print('Cholesky decomposition failed, switching to SVD instead.')
        U, S, Vh = np.linalg.svd(K)
        V = Vh.T
        S_diag = np.diag(S)
        S_invert = 1/S_diag
        K_inverse = V @ S_invert @ U.T

    return K_inverse


def get_kernel(gp, r, u, regularise=False):
    feature_weights = gp.feature_weights
    noise_var = gp.noise_var
    rbf_var = gp.rbf_var

    def kernel_function(x, y, regularise):
        M = x[:, np.newaxis] - y[np.newaxis, :]
        diag_matrix = np.diag(feature_weights)

        if regularise:
            assert x.shape == y.shape
            noise_const = np.exp(-0.5 * noise_var * np.trace(diag_matrix))
            noise_matrix = noise_const * np.ones((x.shape[0], x.shape[0])) + (1 - noise_const) * np.eye(x.shape[0])
            dist_x_y = np.sum(x**2, axis=1) + np.sum(y**2, axis=1).T - 2 * x @ y.T
            dist_x_y = np.maximum(dist_x_y, 0)
            kernel_matrix = rbf_var * np.exp(-0.5 * dist_x_y) * noise_matrix
            #d_uu = np.einsum("imk,kl,lmn->in", M, diag_matrix, M.T)
            #d_uu = np.maximum(d_uu, 0)
            #kernel_matrix = rbf_var * np.exp(-0.5 * d_uu - regulariser)

        else:
            dist_x_y = np.sum(x ** 2, axis=1) + np.sum(y ** 2, axis=1).T - 2 * x @ y.T
            dist_x_y = np.maximum(dist_x_y, 0)
            kernel_matrix = rbf_var * np.exp(-0.5 * dist_x_y)
        return kernel_matrix

    kernel_matrix = kernel_function(r, u, regularise=regularise)
    return kernel_matrix


def get_hp_likelihood(hp, prior, prior_wt, gp, transformation):
    # Compute prior likelihood of hyperparameters.
    hp = hp_transform(hp, transformation, 0)
    # Transform.
    if prior == 'g0':
        # Mean - 0 Gaussian.
        val = -0.5 * np.sum(prior_wt * hp.T**2)
    elif prior == 'gamma':
        # Gamma prior.
        alpha = gp.gamma_shape - 1
        beta = prior_wt
        val = np.sum(alpha @ np.log(hp.T) - beta * hp.T)
    elif prior == 'logsparsity':
        # Logarithmic sparsity penalty.
        val = -np.sum(np.log(prior_wt * hp.T + 1))
    else:
        val = 0
    return val


def pack_parameters(gp, r):
    # Place parameters from specified GP into parameter vector.
    # First, add the reward parameters.
    x = r
    # Next, add ARD kernel parameters.
    x = np.concatenate((x, gp.feature_weights))
    # Add noise hyperparameter if we are learning it.
    if gp.learn_noise:
        x = np.concatenate((x, gp.noise_var))
    # Add RBF variance hyperparameter if we are learning it.
    if gp.learn_rbf:
        x = np.concatenate((x, gp.rbf_var))
    return x


def unpack_parameters(gp, x):
    # Place parameters from parameter vector into GP.
    # Count the last index read.
    last_index = len(x) - 1
    # Read RBF variance hyperparameter if we are learning it.
    if gp.learn_rbf:
        gp.rbf_var = np.array([x[last_index]])
        last_index = last_index - 1
    # Read noise hyperparameter if we are learning it.
    if gp.learn_noise:
        gp.noise_var = np.array([x[last_index]])
        last_index = last_index - 1
    # Read ARD kernel parameters.
    last_index = last_index + 1
    index = last_index - len(gp.feature_weights)
    gp.feature_weights = x[index:last_index].T
    last_index = last_index - len(gp.feature_weights)
    # Read reward parameters.
    r = x[:last_index]
    return gp, r


def hp_transform(hp, transformation, mode):
    if mode == 0:  # Transform from optimisation mode to actual value
        if transformation == 'quad':
            hp = hp**2
        elif transformation == 'exp':
            hp = np.exp(hp)
        elif transformation == 'sig':
            hp = 1./(np.exp(-hp)+1)
        else:
            hp = hp
    if mode == 1:  # Transform from actual value to optimisation mode
        if transformation == 'quad':
            hp = np.sqrt(hp)
        elif transformation == 'exp':
            hp = np.log(hp)
        elif transformation == 'sig':
            hp = -np.log((1./hp)-1)
        else:
            hp = hp
    return hp
