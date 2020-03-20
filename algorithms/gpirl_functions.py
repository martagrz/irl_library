import numpy as np


def invert_kernel(K, y):
    # Safe inversion of kernel matrix that uses SVD decomposition if Cholesky fails.
    # First, try Cholesky.
    try:
        lower_cholesky = np.linalg.cholesky(K).T
        value = np.linalg.solve(lower_cholesky, y)
        alpha = np.linalg.solve(lower_cholesky.conj().T, value)
        half_log_det = np.sum(np.log(np.diag(lower_cholesky)))
        value =  np.linalg.solve(lower_cholesky, np.eye(K.shape[0]))
        K_inverse = np.linalg.solve(lower_cholesky.conj().T, value)

    except:
        print('Cholesky failed, switching to SVD')
        # Must do full SVD decomposition.
        U, S, Vh = np.linalg.svd(K)
        V = Vh.T
        S_diag = np.diag(S)
        S_inverse = np.diag(1 / S_diag)
        alpha = V @ (S_inverse @ (U.T @ y))
        half_log_det = 0.5 * np.sum(np.log(S_diag))
        K_inverse = V @ S_inverse @ U.T

    return alpha, half_log_det, K_inverse


def get_kernel(gp, y, Xstar=None):
    # Optimized kernel computation function for DC mode GPIRL.
    # Constants.
    dims = gp.inv_widths.shape[1]
    n = gp.inducing_points.shape[0]
    # Undo transforms.
    inv_widths = hyper_parameter_transform(gp.inv_widths, [], gp.ard_xform, 1)  # This is \Lambda
    noise_var = hyper_parameter_transform(gp.noise_var, [], gp.noise_xform, 1)  # This is 2\sigma ^ 2
    rbf_var = hyper_parameter_transform(gp.rbf_var, [], gp.rbf_xform, 1)  # This is \beta
    # inv_widths = np.min(inv_widths, int(1e100))  # Prevent overflow.
    # Compute scales.
    iw_sqrt = np.sqrt(inv_widths)
    # Scale positions in feature space.
    X_u_warped = gp.inducing_points
    X_f_warped = gp.X
    X_s_warped = Xstar
    X_u_scaled = iw_sqrt * X_u_warped
    X_f_scaled = iw_sqrt * X_f_warped

    # Construct noise matrix.
    mask_mat = np.ones(n) - np.eye(n)
    # Noise is uniform.
    noise_const = np.exp(-0.5 * noise_var * np.sum(inv_widths))
    noise_matrix = noise_const * np.ones(n) + (1 - noise_const) * np.eye(n)

    # Compute K_uu matrix.
    d_uu = np.sum(X_u_scaled**2, axis=1) + np.sum(X_u_scaled**2, axis=1).T - 2*(X_u_scaled @ X_u_scaled.T)
    # d_uu = np.max(d_uu, 0)
    K_uu = rbf_var * np.exp(-0.5 * d_uu) * noise_matrix

    if Xstar is not None:
        # Use Xstar to compute K_uf matrix.
        X_s_scaled = iw_sqrt * X_s_warped
        d_uf = np.sum(X_u_scaled**2, axis=1) + np.sum(X_s_scaled**2, axis=1).T - 2*(X_u_scaled @ X_s_scaled.T)
        # d_uf = np.max(d_uf, 0)
        K_uf = noise_const * rbf_var * np.exp(-0.5 * d_uf)
    else:
        # Compute K_uf matrix.
        d_uf = np.sum(X_u_scaled**2, axis=1) + np.sum(X_f_scaled ** 2, axis=1).T - 2 * (X_u_scaled @ X_f_scaled.T)
        # d_uf = np.max(d_uf, 0)
        # print('duf', d_uf.shape)
        K_uf = noise_const * rbf_var * np.exp(-0.5 * d_uf)

    # Invert the kernel matrix. - maybe make into try?
    alpha, logDetAndPPrior, invK = invert_kernel(K_uu, y)
    K_ufKinv = K_uf.T @ invK

    # Add hyperparameter prior term which penalizes high partial correlation between inducing points.
    logDetAndPPrior = logDetAndPPrior + 0.5 * np.sum(np.sum(invK**2))

    # Compute gradients.
    hp_cnt = inv_widths.shape[1] + gp.learn_noise + gp.learn_rbf
    dhp = np.zeros((hp_cnt, 1))
    dhpdr = np.zeros((hp_cnt, gp.X.shape[0]))

    # Pre - compute common matrices.
    inmat = (0.5 * 4 * np.linalg.matrix_power(invK, 3) + alpha * alpha.T - invK).T
    iwmat = inmat * K_uu

    # Compute gradient of inverse widths
    for i in np.arange(dims-1):
        du = X_u_warped[:, i]**2 + (X_u_warped[:, i]**2).T - 2*(X_u_warped[:, i] @ X_u_warped[:, i].T)
        df = X_u_warped[:, i]**2 + (X_f_warped[:, i]**2).T - 2*(X_u_warped[:, i] @ X_f_warped[:, i].T)
        # Noise is uniform.
        df = df + noise_var
        du = du + noise_var * mask_mat
        # Compute gradient with respect to length-scales.
        dhp[i, 0] = -0.25 * np.sum(np.sum(iwmat * du))
        # Compute Jacobian of reward with respect to length-scales.
        # This is the component of the Jacobian from dK_uf'*alpha.
        dhpdr[i, :] = -0.5 * np.sum((df*K_uf) * alpha, axis=0)
        #  This is the component of the Jacobian from K_uf'*Kinv*dK*alpha.
        dhpdr[i, :] = dhpdr[i, :] + 0.5 * (K_ufKinv @ np.sum((du*K_uu)*alpha.T, axis=1)).T

    idx = dims

    # Compute gradient of variances.
    if gp.learn_noise:
        # Compute gradient.
        dhp[idx, 0] = -0.25*np.sum(inv_widths)*np.sum(np.sum(iwmat*mask_mat))
        # Compute reward Jacobian.
        # This is the component of the Jacobian from dK_uf'*alpha.
        dhpdr[idx, :] = -0.5*np.sum(inv_widths) * np.sum(K_uf*alpha, axis=0)
        # This is the component of the Jacobian from K_uf'*Kinv*dK*alpha.
        dhpdr[idx, :] = dhpdr[idx, :] + 0.5*np.sum(inv_widths)*(K_ufKinv@np.sum(K_uu*mask_mat*alpha.T, axis=1)).T
        idx = idx+1
    if gp.learn_rbf:
        # Compute gradient.
        dhp[idx, 0] = (0.5/rbf_var)*np.sum(np.sum(iwmat))
        # Compute reward Jacobian.
        # This is the component of the Jacobian from dK_uf'*alpha.
        dhpdr[idx, :] = (1/rbf_var)*np.sum(K_uf*alpha, axis=0)
        # This is the component of the Jacobian from K_uf'*Kinv*dK*alpha.
        dhpdr[idx, :] = dhpdr[idx, :] - (1/rbf_var)*(K_ufKinv@np.sum(K_uu*alpha.T, axis=1)).T
        idx = idx+1

    # Transform gradients.
    value = hyper_parameter_transform(gp.inv_widths, dhp[:dims, :], gp.ard_xform, 2)
    dhp[:dims, :] = value
    dhpdr[:dims, :] = hyper_parameter_transform(gp.inv_widths, dhpdr[:dims, :], gp.ard_xform, 2)
    idx = dims
    if gp.learn_noise:
        dhp[idx, :] = hyper_parameter_transform(gp.noise_var, dhp[idx, :], gp.noise_xform, 2)
        dhpdr[idx, :] = hyper_parameter_transform(gp.noise_var, dhpdr[idx, :], gp.noise_xform, 2)
        idx = idx+1
    if gp.learn_rbf:
        dhp[idx, :] = hyper_parameter_transform(gp.rbf_var, dhp[idx, :], gp.rbf_xform, 2)
        dhpdr[idx, :] = hyper_parameter_transform(gp.rbf_var, dhpdr[idx, :], gp.rbf_xform, 2)
        idx = idx+1

    return K_uf, logDetAndPPrior, alpha, invK, dhp, dhpdr


def get_hp_prior(hp, prior, prior_wt, xform, gp):
    # Compute prior likelihood of hyperparameters.
    # Transform.
    hp = hyper_parameter_transform(hp, [], xform, 1)
    # Make sure we have enough weights.
    if hp.ndim > 1:
        if type(prior_wt) == int:
            prior_wt = np.tile(prior_wt, (hp.shape[1], 1))
        elif type(prior_wt) != int and len(prior_wt) != len(hp):
            prior_wt = np.tile(prior_wt, (hp.shape[1], 1))

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
        val = -np.sum(np.log(prior_wt * hp.T + 1), axis=0)
    else:
        val = 0
    return val


def get_hp_gradient(hp, prior, prior_wt, xform, gp):
    # Compute prior likelihood gradient of hyperparameters.
    # Transform.
    orig_hp = hp
    hp = hyper_parameter_transform(hp, [], xform, 1)
    # Make sure we have enough weights.
    if hp.ndim > 1:
        if type(prior_wt) == int:
            prior_wt = np.tile(prior_wt, (hp.shape[1], 1))
        elif type(prior_wt) != int and len(prior_wt) != len(hp):
            prior_wt = np.tile(prior_wt, (hp.shape[1], 1))
    if prior == 'g0':
        # Mean - 0 Gaussian.
        dp = -prior_wt * hp.T
    elif prior == 'gamma':
        # Gamma prior.
        alpha = gp.gamma_shape - 1
        beta = prior_wt
        dp = (alpha / hp.T) - beta
    elif prior == 'logsparsity':
        # Logarithmic sparsity penalty.
        dp = - prior_wt * (1 / (prior_wt * hp.T + 1))
    else:
        dp = np.zeros((len(hp), 1))
    # Transform back.
    dp = hyper_parameter_transform(orig_hp, dp, xform, 2)
    return dp


def pack_parameters(gp, r):
    # Place parameters from specified GP into parameter vector.
    # First, add the reward parameters.
    x = r
    # Next, add ARD kernel parameters.
    x = np.vstack((x, gp.inv_widths.T))
    # Add noise hyperparameter if we are learning it.
    if gp.learn_noise:
        x = np.vstack((x, gp.noise_var))
    # Add RBF variance hyperparameter if we are learning it.
    if gp.learn_rbf:
        x = np.vstack((x, gp.rbf_var))
    return x


def unpack_parameters(gp, x):
    # Place parameters from parameter vector into GP.
    # Count the last index read.
    last_index = len(x) - 1
    # Read RBF variance hyperparameter if we are learning it.
    if gp.learn_rbf:
        gp.rbf_var = x[last_index]
        last_index = last_index - 1
    # Read noise hyperparameter if we are learning it.
    if gp.learn_noise:
        gp.noise_var = x[last_index]
        last_index = last_index - 1
    # Read ARD kernel parameters.
    last_index = last_index + 1
    index = last_index - gp.inv_widths.shape[1]
    gp.inv_widths = x[index:last_index].T
    last_index = last_index - gp.inv_widths.shape[1]
    # Read reward parameters.
    r = x[:last_index]
    return gp, r


def hyper_parameter_transform(hp, grad, xform, mode):
    if mode == 1:
        # Transform from optimization mode to actual value.
        if xform == 'quad':
            hp = hp ** 2
        elif xform == 'exp':
            hp = np.exp(hp)
        elif xform == 'sig':
            hp = 1 / (np.exp(-hp) + 1)
        else:
            hp = hp
    elif mode == 2:
        # Transform derivative.
        if xform == 'quad':
            hp = 2 * hp.conj().T * grad
        elif xform == 'exp':
            hp = np.exp(hp.conj().T) * grad
            #hp[grad == 0] = 0
        elif xform == 'sig':
            ex_php = np.exp(-hp.conj().T)
            hp = (ex_php / ((ex_php + 1)**2)) * grad
        else:
            hp = grad
    elif mode == 3:
        # Actual value to optimization mode.
        if xform == 'quad':
            hp = np.sqrt(hp)
        elif xform == 'exp':
            hp = np.log(hp)
        elif xform == 'sig':
            hp = -np.log((1 / hp) - 1)
        else:
            hp = hp
    elif mode == 4:
        # Clamp.
        if xform == 'quad':
            hp = np.min(hp, np.sqrt(grad))
        elif xform == 'exp':
            hp = np.min(hp, np.log(grad))
        elif xform == 'sig':
            hp = hp  # Sigmoid never gets large.
        else:
            hp = hp
    return hp
