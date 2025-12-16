import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.special import gamma, gammainc, gammaincc, gammaln, loggamma, kv
from scipy.integrate import quad
from scipy.optimize import brentq


def moment_magnitude(seismic_moment):
    """
    Calculate the moment magnitude of an earthquake from its moment according to Hanks and Kanamori (1979).

    Parameters
    ----------
    moment : float
        The moment of the earthquake.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research: Solid Earth, 84(B5), 2348-2350.
    """
    # moment = 10**(1.5 * moment_magnitude + 9.05)

    return (np.log10(seismic_moment) - 9.05) / 1.5


def seismic_moment(moment_magnitude):
    """
    Calculate the moment of an earthquake from its magnitude according to Hanks and Kanamori (1979).

    Parameters
    ----------
    moment_magnitude : float
        The moment magnitude of the earthquake.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research: Solid Earth, 84(B5), 2348-2350.
    """

    return 10 ** (1.5 * moment_magnitude + 9.05)



def plot_distr(mrange, poe, ax=None, **kwargs):
    """
    Create FMD plot
    :param mrange: Magnitudes (discretized range)
    :param poe: Probability of exceedance
    :param ax: (Optional) Plot on these axes instead of general plt-call
    :param kwargs: (Optional) Keyword arguments to pass on to plotting function
    :return: None
    """
    if ax is not None:
        ax.semilogy(mrange, poe, **kwargs)
    else:
        plt.semilogy(mrange, poe, **kwargs)


def plot_catalogue(catalogue, mrange, ax=None, num=False, **kwargs):
    """
    Plot a cumulative view of a catalogue
    :param catalogue: magnitudes of eqs in catalogue
    :param mrange: Magnitudes (discretized range)
    :param ax: (Optional) Plot on these axes instead of general plt-call
    :param num: (Optional) If True, plot number of events instead of probability of exceedance
    :param kwargs: (Optional) Keyword arguments to pass on to plotting function
    :return: None
    """
    cat_plot = np.array([np.sum(catalogue >= m) / len(catalogue) for m in mrange])
    if num:
        cat_plot = cat_plot * len(catalogue)
    if ax is not None:
        ax.step(mrange, cat_plot, where="mid", **kwargs)
    else:
        plt.step(mrange, cat_plot, where="mid", **kwargs)



def theta_from_q(q, n):
    return - n * np.log(q)

def pdf_naive(y, n, q):
    theta = theta_from_q(q, n)
    z = -np.log(y)           # z > 0
    # log pref = n*log(theta) - ln Gamma(n)
    log_pref = n * np.log(theta) - gammaln(n)
    # logpdf = log_pref - (n+1)*log(z) - theta/z + z
    logpdf = log_pref - (n + 1) * np.log(z) - theta / z + z
    # return pdf
    return np.exp(logpdf)

def cdf_naive(y, n, q):
    theta = theta_from_q(q, n)
    z = -np.log(y)
    return 1.0 - gammaincc(n, theta / z)   # gammaincc is regularized upper gamma Q

def cdf_corr_naive(y, n, q):
    L = -np.log(q)
    z = -np.log(y)
    arg = (n - 1) * L / z
    return 1.0 - gammaincc(n, arg)

def pdf_unbiased(y, n, q):
    L = -np.log(q)
    u = y**(1.0/(n-1))
    one_minus_u = 1.0 - u
    log_pref = n * np.log(L) - gammaln(n) - np.log(n-1)
    log_y_term = (1.0/(n-1) - 1.0) * np.log(y)
    log_one_minus = - (n + 1.0) * np.log(one_minus_u)
    exp_term = - L / one_minus_u
    logpdf = log_pref + log_y_term + log_one_minus + exp_term
    return np.exp(logpdf)

def cdf_unbiased(y, n, q):
    L = -np.log(q)
    u = y**(1.0/(n-1))
    s0 = L / (1.0 - u)
    return gammainc(n, s0)


def pdf_jeffreys(y, n, q):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    # valid mask
    mask = (y > 0.0) & (y < 1.0)
    if not np.any(mask):
        return out

    yy = y[mask]
    L = -np.log(q)  

    u = np.exp(np.log(yy) / n)

    denom = 1.0 - u
    denom = np.maximum(denom, 1e-300)

    w = (L * u) / denom
    log_pref = n * np.log(L) - np.log(n) - gammaln(n)
    log_one_minus = -(n + 1.0) * np.log(denom)

    exp_term = -w
    logpdf = log_pref + log_one_minus + exp_term
    out_vals = np.exp(logpdf)

    out[mask] = out_vals
    return out


def cdf_jeffreys(y, n, q):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    out[y >= 1.0] = 1.0
    mask = (y > 0.0) & (y < 1.0)
    if not np.any(mask):
        return out

    yy = y[mask]
    L = -np.log(q)

    u = np.exp(np.log(yy) / n)
    denom = 1.0 - u
    denom = np.maximum(denom, 1e-300)

    w = (L * u) / denom
    out[mask] = gammainc(n, w)

    return out


def mean_var_from_cdf(cdf_func, n, q,
                               tmin_override=None,
                               tpad=50.0,
                               epsrel=1e-10,
                               limit=800):

    # choose tmin (lower limit of log-space)
    # base on q: mass of distribution typically around q; ensure we go well below it
    if tmin_override is None:
        if q <= 0.0:
            t_q = -50.0
        else:
            t_q = math.log(q)
        # set tmin such that y_min = exp(tmin) is extremely small relative to q
        # e.g. y_min = q * exp(-tpad) => tmin = log(q) - tpad
        tmin = max(t_q - tpad, -745.0)
    else:
        tmin = float(tmin_override)
    tmax = 0.0

    # integrands in t-space
    def integrand_mean(t):
        y = math.exp(t)
        # compute 1 - F(y) carefully
        Fy = float(cdf_func(y, n, q))
        tail = 1.0 - Fy
        # guard tiny negative due to rounding
        if tail < 0.0:
            tail = 0.0
        return tail * y

    def integrand_E2term(t):
        y = math.exp(t)
        Fy = float(cdf_func(y, n, q))
        tail = 1.0 - Fy
        if tail < 0.0:
            tail = 0.0
        return 2.0 * tail * (y * y)   # 2*y * (1-F(y)) * dy/dt where dy/dt = y

    mean, err_mean = quad(integrand_mean, tmin, tmax, epsrel=epsrel, epsabs=0.0, limit=limit)
    E2, err_E2 = quad(integrand_E2term, tmin, tmax, epsrel=epsrel, epsabs=0.0, limit=limit)
    var = E2 - mean * mean

    # diagnostics and fallbacks
    diag = {'tmin': tmin, 'err_mean': err_mean, 'err_E2': err_E2,
            'mean_raw': mean, 'E2_raw': E2, 'var_raw': var}

    tol_rel = 1e-12
    tol_abs = 1e-20
    if var < 0.0:
        if var >= -max(tol_abs, tol_rel * mean * mean):
            var = 0.0
            diag['clamped'] = True
        else:
            # large negative => try expanding integration domain (more negative tmin)
            tmin2 = max(tmin - 4.0 * tpad, -3000.0)
            try:
                mean2, err_mean2 = quad(integrand_mean, tmin2, tmax, epsrel=epsrel/10.0, epsabs=0.0, limit=limit*2)
                E22, err_E22    = quad(integrand_E2term, tmin2, tmax, epsrel=epsrel/10.0, epsabs=0.0, limit=limit*2)
                var2 = E22 - mean2 * mean2
                diag.update({'recompute_tmin2': tmin2,
                             'mean2': mean2, 'E22': E22, 'var2': var2,
                             'err_mean2': err_mean2, 'err_E22': err_E22})
                if var2 >= -max(tol_abs, tol_rel * mean2 * mean2):
                    mean, E2, var = mean2, E22, max(0.0, var2)
                else:
                    # still bad: raise an informative RuntimeError
                    raise RuntimeError("Numerical integration unstable: E2 < mean^2 even after expanding tmin.")
            except Exception as e:
                # return diagnostics and raise
                diag['error'] = str(e)
                raise RuntimeError("mean/var integration unstable; see diag") from e

    return float(mean), float(var), diag

# robust percentile inversion using analytic CDF function
def percentile_from_cdf_scalar(cdf_func, p, n, q):
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # bracket
    y_lo = 1e-300
    y_hi = 1.0 - 1e-16

    def g(y):
        return float(cdf_func(y, n, q) - p)

    if g(y_lo) >= 0:
        return y_lo
    if g(y_hi) <= 0:
        return y_hi

    root = brentq(g, y_lo, y_hi, xtol=1e-14, rtol=1e-12, maxiter=200)
    return float(root)


def summary_naive(q, n):
    mean, var, _ = mean_var_from_cdf(cdf_naive, n, q)
    p2_5 = percentile_from_cdf_scalar(cdf_naive, 0.025, n, q)
    p50  = percentile_from_cdf_scalar(cdf_naive, 0.50, n, q)
    p97_5= percentile_from_cdf_scalar(cdf_naive, 0.975, n, q)
    return {'mean': mean, 'var': var, 'p2.5': p2_5, 'p50': p50, 'p97.5': p97_5}


def summary_corrected_naive(q, n):
    mean, var, _ = mean_var_from_cdf(cdf_corr_naive, n, q)
    p2_5 = percentile_from_cdf_scalar(cdf_corr_naive, 0.025, n, q)
    p50  = percentile_from_cdf_scalar(cdf_corr_naive, 0.50, n, q)
    p97_5= percentile_from_cdf_scalar(cdf_corr_naive, 0.975, n, q)
    return {'mean': mean, 'var': var, 'p2.5': p2_5, 'p50': p50, 'p97.5': p97_5}

def summary_jeffreys(q, n):
    mean, var, _ = mean_var_from_cdf(cdf_jeffreys, n, q)
    p2_5 = percentile_from_cdf_scalar(cdf_jeffreys, 0.025, n, q)
    p50  = percentile_from_cdf_scalar(cdf_jeffreys, 0.50, n, q)
    p97_5= percentile_from_cdf_scalar(cdf_jeffreys, 0.975, n, q)
    return {'mean': mean, 'var': var, 'p2.5': p2_5, 'p50': p50, 'p97.5': p97_5}

def summary_unbiased(q, n):
    mean, var, _ = mean_var_from_cdf(cdf_unbiased, n, q)
    p2_5 = percentile_from_cdf_scalar(cdf_unbiased, 0.025, n, q)
    p50  = percentile_from_cdf_scalar(cdf_unbiased, 0.50, n, q)
    p97_5= percentile_from_cdf_scalar(cdf_unbiased, 0.975, n, q)
    return {'mean': mean, 'var': var, 'p2.5': p2_5, 'p50': p50, 'p97.5': p97_5}


def gamma_pdf(x, alpha, lamb):
    return (lamb**alpha) * (x**(alpha-1)) * np.exp(-lamb*x) / gamma(alpha)

def norm_pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( -0.5 * ((x - mu)/sigma)**2 )

def integrate_stable_1d(log_integrand, s):
    max_log = np.max(log_integrand)
    return np.trapezoid(np.exp(log_integrand - max_log), s) * np.exp(max_log)

def E_generic_stable(n, beta, x_array, log_g_func):
    s = np.linspace(1e-10, 10 * n / beta, 8000)
    log_gamma_n = loggamma(n)
    log_f_S = n * np.log(beta) - log_gamma_n + (n - 1) * np.log(s) - beta * s
    results = np.empty_like(x_array, dtype=float)
    
    for i, x in enumerate(x_array):
        log_g = log_g_func(s, x, n)
        log_integrand = log_f_S + log_g
        results[i] = integrate_stable_1d(log_integrand, s)
    return results

def log_g_naive(s, x, n):
    return -n * x / s

def log_g_corrected(s, x, n):
    return -(n - 1) * x / s

def log_g_unbiased(s, x, n):
    log_g = np.full_like(s, -np.inf)
    mask = s > x
    log_g[mask] = (n - 1) * np.log(1 - x / s[mask])
    return log_g

def log_g_jeffreys(s, x, n):
    return n * (np.log(s) - np.log(s + x))

def E_all(n, beta, x_array):
    return {
        "naive": E_generic_stable(n, beta, x_array, log_g_naive),
        "corrected": E_generic_stable(n, beta, x_array, log_g_corrected),
        "unbiased": E_generic_stable(n, beta, x_array, log_g_unbiased),
        "jeffreys": E_generic_stable(n, beta, x_array, log_g_jeffreys),
    }


def _expect_exp_minus_c_over_S(x, n, beta, c_factor):
    """
    Compute E[exp(-c_factor * n * x / S)] for S ~ Gamma(n, rate=beta),
    using the closed-form Bessel K expression.

    c_factor = 1  -> estimator (i)
    c_factor = 1-1/n -> estimator (ii) with (n-1)x, but we’ll pass it explicitly.
    More generally, we pass c = c_factor * n * x.
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)

    # x <= 0 -> exp(0) = 1, expectation = 1
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    c = c_factor * n * xm       # c = nx or (n-1)x, etc.
    z = 2.0 * np.sqrt(beta * c) # argument of K_n

    log_pref = (
        np.log(2.0)
        + 0.5 * n * (np.log(beta) + np.log(c))
        - loggamma(n)
    )

    Em = np.exp(log_pref) * kv(n, z)

    out[mask] = Em
    return out


def get_var_naive(x, n, beta):
    x = np.asarray(x, dtype=float)
    EF = _expect_exp_minus_c_over_S(x, n, beta, c_factor=1.0)
    EF_sq = _expect_exp_minus_c_over_S(x, n, beta, c_factor=2.0)
    var = EF_sq - EF**2

    return np.maximum(var, 0.0)


def get_var_corrected_naive(x, n, beta):
    x = np.asarray(x, dtype=float)
    
    if n == 1:
        return np.zeros_like(x, dtype=float)
    
    c_fac = (n - 1.0) / n
    EF  = _expect_exp_minus_c_over_S(x, n, beta, c_factor=c_fac)
    EF_sq = _expect_exp_minus_c_over_S(x, n, beta, c_factor=2*c_fac)

    var = EF_sq - EF**2
    return np.maximum(var, 0.0)

def get_var_jeffreys(x, n, beta, num_u=2000, u_min=1e-6, u_max=1-1e-6):
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x_flat = x.ravel()

    var = np.zeros_like(x_flat, dtype=float)

    mask = x_flat > 0
    if not np.any(mask):
        return var.reshape(shape)

    xs = x_flat[mask]

    u = np.linspace(u_min, u_max, num_u)
    log_dt_du = -2.0 * np.log1p(-u)
    t = u / (1.0 - u)

    log_t      = np.log(t)
    logGamma_n = float(loggamma(n))


    T = t[None, :]
    log_T = log_t[None, :]

    beta_x = (beta * xs)[:, None]
    log_dt_du_row = log_dt_du[None, :]

    log_int_E = (
        -T
        + (n - 1.0) * log_T
        + n * (log_T - np.log(T + beta_x))
        + log_dt_du_row
        - logGamma_n
    )

    C_E = np.max(log_int_E, axis=1, keepdims=True)
    integrand_E = np.exp(log_int_E - C_E)
    I_E = np.trapezoid(integrand_E, u, axis=1)
    EF = np.exp(C_E[:, 0]) * I_E

    log_int_E2 = (
        -T
        + (n - 1.0) * log_T
        + 2.0 * n * (log_T - np.log(T + beta_x))
        + log_dt_du_row
        - logGamma_n
    )

    C_E2 = np.max(log_int_E2, axis=1, keepdims=True)
    integrand_E2 = np.exp(log_int_E2 - C_E2)
    I_E2 = np.trapezoid(integrand_E2, u, axis=1)
    EF_sq = np.exp(C_E2[:, 0]) * I_E2

    var_vals = np.maximum(EF_sq - EF**2, 0.0)
    var[mask] = var_vals

    return var.reshape(shape)


def get_var_unbiased(x, n, beta, num_u=2000, u_min=1e-6, u_max=1-1e-6):
    x = np.asarray(x, dtype=float)
    out_shape = x.shape
    x_flat = x.ravel()

    var = np.zeros_like(x_flat, dtype=float)

    mask = x_flat > 0
    if not np.any(mask):
        return var.reshape(out_shape)

    xs = x_flat[mask]
    k = xs.size

    u = np.linspace(u_min, u_max, num_u)
    log_dt_du = -2.0 * np.log1p(-u) 


    bx = (beta * xs)[:, None]
    U = u[None, :]
    T = bx + U / (1.0 - U) 
    log_T = np.log(T)


    logGamma_n = float(loggamma(n))
    ratio = bx / T
    ratio = np.minimum(ratio, 1.0 - 1e-15)
    log_one_minus_ratio = np.log1p(-ratio)

    log_dt_du_row = log_dt_du[None, :]

    log_int_E2 = (
        -T
        + (n - 1.0) * log_T
        + 2.0 * (n - 1.0) * log_one_minus_ratio
        + log_dt_du_row
        - logGamma_n
    )

    C = np.max(log_int_E2, axis=1, keepdims=True)
    integrand = np.exp(log_int_E2 - C)
    I = np.trapezoid(integrand, u, axis=1)
    EF_sq = np.exp(C[:, 0]) * I


    EF = np.exp(-beta * xs)
    var_vals = np.maximum(EF_sq - EF**2, 0.0)

    var[mask] = var_vals
    return var.reshape(out_shape)