import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, random
from functools import partial
from jax.lax import fori_loop
import numpy as np
import yaml

EPSILON = 1e-12

### JAX
# Unscented Kalman Filter points
def sigma_points_jax(mean, cov, lam=3, alpha=1e-3, beta=2):
    """
    Compute the sigma points for the unscented transform for an *input*
    Gaussian with mean mu and covariance sigma.
    Returns a (D, 2D+1) array, where D is the dimensionality of the state space.    
    """
    n = len(mean)    
    sqrt_sigma = jnp.linalg.cholesky((n + lam) * cov)   # needs to be Hermetian positive definite
    sigma_points = jnp.vstack([mean, mean + sqrt_sigma, mean - sqrt_sigma])
    return sigma_points.T

def unscented_weights_jax(n, lam=3, alpha=1e-3, beta=2):
    """
    Compute the weights for the unscented transform.
    Note: these depend on the dimensionality of the state space, but
    not on the function being applied or the current state,
    and so can be pre-computed if you want.
    """
    weights_mean = jnp.ones(2 * n + 1) / (2 * (n + lam))
    weights_cov = jnp.array(weights_mean)
    weights_mean = weights_mean.at[0].set(lam / (n + lam) ) #weights_mean[0] = lam / (n + lam) 
    weights_cov = weights_cov.at[0].set(weights_mean[0] + (1 - alpha**2 + beta))  #weights_cov[0] = weights_mean[0] + (1 - alpha**2 + beta)
    return weights_mean, weights_cov

def unscented_jax_n(mean, cov, n, fn, kappa=0, alpha=1e-3, beta=2):
    """
    Take a Gaussian, parameterised by mean and covariance, and apply a nonlinear function to it.
    Return the mean and covariance of the resulting Gaussian.
    fn(x) takes a (N,D) array of N samples of dimension D and returns a (N,D) array of N samples of dimension D.
    lam, alpha, beta are "fiddle factors" and don't usually need much tuning.
    """    
    lam = alpha**2 * (n + kappa) - n
    # get the sigma points
    points = sigma_points_jax(mean, cov, lam, alpha, beta)
    # apply fn
    transformed = jnp.apply_along_axis(fn, 0, points)
    m = transformed.shape[0]
    # get the weights
    weights_mean, weights_cov = unscented_weights_jax(n, lam, alpha, beta)

    # re-estimate the mean and covariance using the weighted samples
    mean_hat = jnp.dot(transformed, weights_mean)

    def body_fun(i, carry):
        carry += weights_cov[i] * \
                jnp.outer(transformed[:, i] - mean_hat,
                            transformed[:, i] - mean_hat)
        return carry
    cov_hat = fori_loop(0, 2 * n + 1, body_fun, jnp.zeros((m, m)))

    # Ensure diagonal is positive
    cov_hat = cov_hat.at[jnp.diag_indices(m)].set(jnp.clip(jnp.diag(cov_hat), min=EPSILON))

    return mean_hat, cov_hat


def unscented(mean, cov, fn, kappa=0, alpha=1e-3, beta=2):
    """
    Take a Gaussian, parameterised by mean and covariance, and apply a nonlinear function to it.
    Return the mean and covariance of the resulting Gaussian.
    fn(x) takes a (N,D) array of N samples of dimension D and returns a (N,E) array of N samples of dimension E.
    lam, alpha, beta are "fiddle factors" and don't usually need much tuning.
    """    
    n = len(mean)
    return unscented_jax_n(mean, cov, n, fn, kappa, alpha, beta)


# Jitting
@partial(jit, static_argnums=(2,3,4,5))
def unscented_jit(mean, cov, fn, kappa=3e6, alpha=1e-3, beta=2):
    return unscented(mean, cov, fn, kappa, alpha, beta)

def softmax_jax(x):
    e = jnp.exp(x - x.max())
    return e / e.sum()

def kl_jax(m1, cov1, m2, cov2):
    m1 = jnp.atleast_1d(m1)
    cov1 = jnp.atleast_2d(cov1)
    m2 = jnp.atleast_1d(m2)
    cov2 = jnp.atleast_2d(cov2)
    if len(m1) == 1:
        return kl_normal_normal(m1, jnp.sqrt(cov1).flatten(), m2, jnp.sqrt(cov2).flatten())
    cov2inv = jnp.linalg.inv(cov2)
    n = m1.shape[0]
    kl = 0.5*(jnp.matmul(jnp.transpose(m2-m1),jnp.matmul(cov2inv,(m2-m1))) + 
            jnp.linalg.trace(jnp.matmul(cov2inv,cov1)) + 
            jnp.log(jnp.linalg.det(cov2)/jnp.linalg.det(cov1)) - 
            n)
    return jnp.squeeze(kl)

def kl_normal_normal(loc1, scale1, loc2, scale2):
    kl = (scale1**2 + (loc1-loc2)**2) / (2*scale2**2) + jnp.log(scale2/scale1) - 0.5 # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return jnp.squeeze(kl)

def generate_n_bit_numbers(n):
    """Generate all possible bit numbers for an n-bit number."""
    for i in range(2**n):  # Iterate over all numbers from 0 to 2^n - 1
        yield format(i, f'0{n}b')  # Convert to binary string with leading zeros

def gen_fixed_plans(a_lims, horizon, dim_action):
    plans = []
    for bit_number in generate_n_bit_numbers(dim_action):
        action = jnp.zeros((dim_action,))
        for j in range(dim_action):
            if bit_number[j] == '0':
                action = action.at[j].set(a_lims[0][j])
            else:
                action = action.at[j].set(a_lims[1][j])
        plans.append(jnp.tile(action, (horizon,1)))
    return jnp.array(plans)


def generate_n_base_m_numbers(n, m):
    """Generate all possible numbers for an n-digit number in base m."""
    def to_base_m(num, base, length):
        """Convert a number to a base-m representation with leading zeros."""
        digits = []
        for _ in range(length):
            digits.append(str(num % base))
            num //= base
        return ''.join(reversed(digits))
    
    for i in range(m**n):  # Iterate over all numbers from 0 to m^n - 1
        yield to_base_m(i, m, n)

def mix_exp(x, min, max):
    res = x.copy()
    res = res.at[x>=0].set(jnp.exp(x[x>=0]*jnp.log(max+1))-1)
    res = res.at[x<0].set(-(jnp.exp(-x[x<0]*jnp.log(-min+1))-1))
    return res

def gen_fixed_plans(a_lims, horizon, dim_action, num_plans, uniform=True):
    plans = []

    num_actions_per_dim = int(num_plans**(1/dim_action))
    if num_actions_per_dim**dim_action != num_plans:
        print(f"Note on use_fixed_plans: Using {num_actions_per_dim**dim_action} different plans instead of {num_plans} for action selection (num actions per dim: {num_actions_per_dim}).")
    
    plan_ids = create_plan_ids(dim_action, num_actions_per_dim)
    actions = create_fixed_actions(a_lims, dim_action, num_actions_per_dim, uniform)

    for plan_id in plan_ids:
        action = jnp.zeros((dim_action,))
        for j in range(dim_action):
            action = action.at[j].set(actions[j][int(plan_id[j])])
        plans.append(jnp.tile(action, (horizon,1)))
    return jnp.array(plans)

def create_plan_ids(dim_action, num_actions_per_dim):
    return _create_plan_ids([[]], dim_action, num_actions_per_dim)

def _create_plan_ids(plans, dim_action, num_actions_per_dim):
    if dim_action > 0:
        new_plans = []
        for plan in plans:
            for i in range(num_actions_per_dim):
                plan_i = plan.copy()
                plan_i.append(i)
                new_plans.append(plan_i)
        return _create_plan_ids(new_plans, dim_action-1, num_actions_per_dim)
    else:
        return plans

def create_fixed_actions(a_lims, dim_action, num_actions_per_dim, uniform=True):
    actions = []
    if uniform:
        for j in range(dim_action):
            a = jnp.linspace(a_lims[0][j], a_lims[1][j], num_actions_per_dim)
            actions.append(a)
    else:
        for j in range(dim_action):
            uniform_distributed_values = jnp.linspace(-1, 1, num_actions_per_dim)
            a = mix_exp(uniform_distributed_values, a_lims[0][j], a_lims[1][j])
            actions.append(a)

    return actions

def refactor_noise_params(noise_params):
    noise_id = 0
    dim_noise = 0
    for (noise_name, noise_setting) in noise_params.items():
        assert 'id' in noise_setting.keys(), f"Noise parameter for '{noise_name}' does not contain an 'id' key defining the index/indices of the corresponding noise."
        try:
            num_noise_values = len(noise_setting['id'])
        except TypeError:
            num_noise_values = 1
        if num_noise_values > 1:
            noise_params[noise_name] = [np.array([index for index in range(noise_id, noise_id + num_noise_values)]), noise_setting['id']]
        else:
            noise_params[noise_name] = [jnp.atleast_1d(noise_id)] + [jnp.atleast_1d(noise_setting['id'])]

        if 'value' in noise_setting.keys():
            # Noise settings for generative process
            if num_noise_values > 1:
                noise_params[noise_name].append(noise_setting['value'])
            else:
                dim_noise += 1
                noise_params[noise_name].append(jnp.atleast_1d(noise_setting['value']))
        dim_noise += num_noise_values
        noise_id += num_noise_values
       
    return noise_params, dim_noise

# Load data from a YAML file
def load_yaml_file(path):
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        print(f"Warning: File not found: {path}.")
        return {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{path}': {e}")