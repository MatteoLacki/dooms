from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

from .fdr import (
    bh_adjust_1d,
    bh_adjust_by_comparison,
    bh_adjust_by_protein,
    bh_adjust_flat_matrix,
    mask_tukey_by_protein,
)
from .masks import presence_masks_from_counts


SQRT2 = np.sqrt(2.0)
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
PI = np.pi
S_NODES, S_WEIGHTS = np.polynomial.legendre.leggauss(64)
Z_NODES, Z_WEIGHTS = np.polynomial.legendre.leggauss(64)
CH_XLEG = np.array(
    [
        0.9815606342467193,
        0.9041172563704749,
        0.7699026741943047,
        0.5873179542866174,
        0.3678314989981802,
        0.1252334085114689,
    ],
    dtype=np.float64,
)
CH_ALEG = np.array(
    [
        0.04717533638651183,
        0.10693932599531843,
        0.16007832854334623,
        0.20316742672306592,
        0.2334925365383548,
        0.24914704581340279,
    ],
    dtype=np.float64,
)
CH_XLEGQ = np.array(
    [
        0.9894009349916499,
        0.9445750230732326,
        0.8656312023878318,
        0.755404408355003,
        0.6178762444026437,
        0.4580167776572274,
        0.2816035507792589,
        0.09501250983763744,
    ],
    dtype=np.float64,
)
CH_ALEGQ = np.array(
    [
        0.027152459411754095,
        0.06225352393864789,
        0.09515851168249278,
        0.12462897125553387,
        0.14959598881657673,
        0.16915651939500254,
        0.1826034150449236,
        0.1894506104550685,
    ],
    dtype=np.float64,
)
STUDENTIZED_R_GRID_SIZE = 4096
STUDENTIZED_R_GRID_MAX = 16.0


@njit(cache=True)
def _accumulate_summaries(protein_codes, condition_codes, intensity, n_proteins, n_conditions):
    counts = np.zeros((n_proteins, n_conditions), np.int64)
    sums = np.zeros((n_proteins, n_conditions), np.float64)
    sums_sq = np.zeros((n_proteins, n_conditions), np.float64)

    for i in range(intensity.size):
        protein = protein_codes[i]
        condition = condition_codes[i]
        value = intensity[i]
        if protein >= 0 and condition >= 0 and np.isfinite(value):
            counts[protein, condition] += 1
            sums[protein, condition] += value
            sums_sq[protein, condition] += value * value

    return counts, sums, sums_sq


@njit(parallel=True, cache=True)
def _anova_kernel(counts, sums, sums_sq):
    n_proteins, n_conditions = counts.shape
    n_total = np.zeros(n_proteins, np.int64)
    k = np.zeros(n_proteins, np.int64)
    ssb = np.full(n_proteins, np.nan)
    ssw = np.full(n_proteins, np.nan)
    df_between = np.full(n_proteins, np.nan)
    df_within = np.full(n_proteins, np.nan)
    msw = np.full(n_proteins, np.nan)
    f_value = np.full(n_proteins, np.nan)

    for protein in prange(n_proteins):
        total_n = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        ssb_temp = 0.0
        condition_count = 0

        for condition in range(n_conditions):
            n = counts[protein, condition]
            if n > 0:
                condition_count += 1
                sx = sums[protein, condition]
                total_n += n
                total_sum += sx
                total_sum_sq += sums_sq[protein, condition]
                ssb_temp += (sx * sx) / n

        n_total[protein] = total_n
        k[protein] = condition_count

        if condition_count == n_conditions and total_n > condition_count:
            protein_ssb = ssb_temp - (total_sum * total_sum) / total_n
            protein_ssw = total_sum_sq - ssb_temp
            db = condition_count - 1
            dw = total_n - condition_count
            msb = protein_ssb / db
            protein_msw = protein_ssw / dw

            ssb[protein] = protein_ssb
            ssw[protein] = protein_ssw
            df_between[protein] = db
            df_within[protein] = dw
            msw[protein] = protein_msw
            if protein_msw != 0.0:
                f_value[protein] = msb / protein_msw

    return n_total, k, ssb, ssw, df_between, df_within, msw, f_value


@njit(parallel=True, cache=True)
def _tukey_q_kernel(counts, sums, msw, k, df_within, pairs):
    n_pairs = pairs.shape[0]
    n_proteins = counts.shape[0]
    q_values = np.full((n_pairs, n_proteins), np.nan)
    pair_k = np.full((n_pairs, n_proteins), np.nan)
    pair_df = np.full((n_pairs, n_proteins), np.nan)

    for flat in prange(n_pairs * n_proteins):
        pair_idx = flat // n_proteins
        protein = flat - pair_idx * n_proteins
        a = pairs[pair_idx, 0]
        b = pairs[pair_idx, 1]
        n_a = counts[protein, a]
        n_b = counts[protein, b]

        if n_a > 0 and n_b > 0 and np.isfinite(msw[protein]) and msw[protein] > 0:
            mean_a = sums[protein, a] / n_a
            mean_b = sums[protein, b] / n_b
            harmonic_mean = (2.0 * n_a * n_b) / (n_a + n_b)
            se = np.sqrt(msw[protein] / harmonic_mean)
            if se > 0.0 and np.isfinite(se):
                q_values[pair_idx, protein] = abs((mean_a - mean_b) / se)
                pair_k[pair_idx, protein] = k[protein]
                pair_df[pair_idx, protein] = df_within[protein]

    return q_values, pair_k, pair_df


@njit(cache=True)
def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))


@njit(cache=True)
def _studentized_range_sf_asymptotic(q, k, z_nodes, z_weights):
    zmax = 8.0
    cdf = 0.0
    for i in range(z_nodes.size):
        z = zmax * z_nodes[i]
        weight = zmax * z_weights[i]
        phi = np.exp(-0.5 * z * z) * INV_SQRT_2PI
        delta = _normal_cdf(z + q) - _normal_cdf(z)
        if delta < 0.0:
            delta = 0.0
        cdf += weight * k * phi * delta ** (k - 1.0)

    sf = 1.0 - cdf
    if sf < 0.0:
        return 0.0
    if sf > 1.0:
        return 1.0
    return sf


@njit(cache=True)
def _studentized_range_sf(q, k, df, s_nodes, s_weights, z_nodes, z_weights):
    if not (np.isfinite(q) and np.isfinite(k) and np.isfinite(df)):
        return np.nan
    if q < 0.0 or k <= 1.0 or df <= 0.0:
        return np.nan
    if df > 1000.0:
        return _studentized_range_sf_asymptotic(q, k, z_nodes, z_weights)

    zmax = 8.0
    width = 10.0 / np.sqrt(2.0 * df)
    if width < 0.6:
        width = 0.6
    lo = 1.0 - width
    if lo < 1e-6:
        lo = 1e-6
    hi = 1.0 + width
    s_mid = 0.5 * (hi + lo)
    s_scale = 0.5 * (hi - lo)
    log_const = (
        np.log(2.0)
        + (df / 2.0) * np.log(df / 2.0)
        - math.lgamma(df / 2.0)
    )

    cdf = 0.0
    for si in range(s_nodes.size):
        s = s_mid + s_scale * s_nodes[si]
        s_weight = s_scale * s_weights[si]
        log_density = log_const + (df - 1.0) * np.log(s) - (df * s * s / 2.0)
        if log_density < -745.0:
            continue
        s_density = np.exp(log_density)

        inner = 0.0
        for zi in range(z_nodes.size):
            z = zmax * z_nodes[zi]
            z_weight = zmax * z_weights[zi]
            phi = np.exp(-0.5 * z * z) * INV_SQRT_2PI
            delta = _normal_cdf(z + q * s) - _normal_cdf(z)
            if delta < 0.0:
                delta = 0.0
            inner += z_weight * phi * delta ** (k - 1.0)

        cdf += s_weight * s_density * k * inner

    sf = 1.0 - cdf
    if sf < 0.0:
        return 0.0
    if sf > 1.0:
        return 1.0
    return sf


@njit(cache=True)
def _studentized_log_cdf_const(k, df):
    return (
        np.log(k)
        + (df / 2.0) * np.log(df)
        - (math.lgamma(df / 2.0) + (df / 2.0 - 1.0) * np.log(2.0))
    )


@njit(cache=True)
def _studentized_cdf_integrand(q, k, df, log_const, z, s):
    delta = _normal_cdf(z + q * s) - _normal_cdf(z)
    if delta <= 0.0:
        return 0.0
    log_terms = (
        log_const
        + (df - 1.0) * np.log(s)
        - (df * s * s / 2.0)
        - 0.5 * z * z
        + np.log(INV_SQRT_2PI)
    )
    if log_terms < -745.0:
        return 0.0
    return np.exp(log_terms) * delta ** (k - 1.0)


@njit(cache=True)
def _studentized_asymptotic_cdf_integrand(q, k, z):
    delta = _normal_cdf(z + q) - _normal_cdf(z)
    if delta <= 0.0:
        return 0.0
    phi = np.exp(-0.5 * z * z) * INV_SQRT_2PI
    return k * phi * delta ** (k - 1.0)


@njit(cache=True)
def _z_eval_transformed(u, q, k, df, log_const, s, asymptotic):
    eps = 1e-12
    if u <= eps or u >= 1.0 - eps:
        return 0.0
    z = math.tan(PI * (u - 0.5))
    jac = PI * (1.0 + z * z)
    if asymptotic:
        return _studentized_asymptotic_cdf_integrand(q, k, z) * jac
    return _studentized_cdf_integrand(q, k, df, log_const, z, s) * jac


@njit(cache=True)
def _adaptive_simpson_z(q, k, df, log_const, s, asymptotic, eps, max_depth):
    a = 1e-10
    b = 1.0 - 1e-10
    c = 0.5 * (a + b)
    fa = _z_eval_transformed(a, q, k, df, log_const, s, asymptotic)
    fb = _z_eval_transformed(b, q, k, df, log_const, s, asymptotic)
    fc = _z_eval_transformed(c, q, k, df, log_const, s, asymptotic)
    whole = (b - a) * (fa + 4.0 * fc + fb) / 6.0
    return _adaptive_simpson_z_rec(
        a, b, fa, fb, fc, whole, q, k, df, log_const, s, asymptotic, eps, max_depth
    )


@njit(cache=True)
def _adaptive_simpson_z_rec(
    a, b, fa, fb, fc, whole, q, k, df, log_const, s, asymptotic, eps, depth
):
    c = 0.5 * (a + b)
    left_mid = 0.5 * (a + c)
    right_mid = 0.5 * (c + b)
    fd = _z_eval_transformed(left_mid, q, k, df, log_const, s, asymptotic)
    fe = _z_eval_transformed(right_mid, q, k, df, log_const, s, asymptotic)
    left = (c - a) * (fa + 4.0 * fd + fc) / 6.0
    right = (b - c) * (fc + 4.0 * fe + fb) / 6.0
    delta = left + right - whole
    if depth <= 0 or abs(delta) <= 15.0 * eps:
        return left + right + delta / 15.0
    return _adaptive_simpson_z_rec(
        a, c, fa, fc, fd, left, q, k, df, log_const, s, asymptotic, eps / 2.0, depth - 1
    ) + _adaptive_simpson_z_rec(
        c, b, fc, fb, fe, right, q, k, df, log_const, s, asymptotic, eps / 2.0, depth - 1
    )


@njit(cache=True)
def _s_eval_transformed(v, q, k, df, log_const, eps_z, max_depth_z):
    eps = 1e-12
    if v <= eps or v >= 1.0 - eps:
        return 0.0
    s = v / (1.0 - v)
    jac = 1.0 / ((1.0 - v) * (1.0 - v))
    return _adaptive_simpson_z(q, k, df, log_const, s, False, eps_z, max_depth_z) * jac


@njit(cache=True)
def _adaptive_simpson_s(q, k, df, log_const, eps_s, eps_z, max_depth_s, max_depth_z):
    a = 1e-10
    b = 1.0 - 1e-10
    c = 0.5 * (a + b)
    fa = _s_eval_transformed(a, q, k, df, log_const, eps_z, max_depth_z)
    fb = _s_eval_transformed(b, q, k, df, log_const, eps_z, max_depth_z)
    fc = _s_eval_transformed(c, q, k, df, log_const, eps_z, max_depth_z)
    whole = (b - a) * (fa + 4.0 * fc + fb) / 6.0
    return _adaptive_simpson_s_rec(
        a, b, fa, fb, fc, whole, q, k, df, log_const, eps_s, eps_z, max_depth_s, max_depth_z
    )


@njit(cache=True)
def _adaptive_simpson_s_rec(
    a, b, fa, fb, fc, whole, q, k, df, log_const, eps_s, eps_z, depth_s, depth_z
):
    c = 0.5 * (a + b)
    left_mid = 0.5 * (a + c)
    right_mid = 0.5 * (c + b)
    fd = _s_eval_transformed(left_mid, q, k, df, log_const, eps_z, depth_z)
    fe = _s_eval_transformed(right_mid, q, k, df, log_const, eps_z, depth_z)
    left = (c - a) * (fa + 4.0 * fd + fc) / 6.0
    right = (b - c) * (fc + 4.0 * fe + fb) / 6.0
    delta = left + right - whole
    if depth_s <= 0 or abs(delta) <= 15.0 * eps_s:
        return left + right + delta / 15.0
    return _adaptive_simpson_s_rec(
        a, c, fa, fc, fd, left, q, k, df, log_const, eps_s / 2.0, eps_z, depth_s - 1, depth_z
    ) + _adaptive_simpson_s_rec(
        c, b, fc, fb, fe, right, q, k, df, log_const, eps_s / 2.0, eps_z, depth_s - 1, depth_z
    )


@njit(cache=True)
def _adaptive_simpson_z_iter(q, k, df, log_const, s, asymptotic, eps, max_depth):
    max_stack = 4096
    stack_a = np.empty(max_stack, np.float64)
    stack_b = np.empty(max_stack, np.float64)
    stack_fa = np.empty(max_stack, np.float64)
    stack_fb = np.empty(max_stack, np.float64)
    stack_fc = np.empty(max_stack, np.float64)
    stack_whole = np.empty(max_stack, np.float64)
    stack_eps = np.empty(max_stack, np.float64)
    stack_depth = np.empty(max_stack, np.int64)

    a = 1e-10
    b = 1.0 - 1e-10
    c = 0.5 * (a + b)
    fa = _z_eval_transformed(a, q, k, df, log_const, s, asymptotic)
    fb = _z_eval_transformed(b, q, k, df, log_const, s, asymptotic)
    fc = _z_eval_transformed(c, q, k, df, log_const, s, asymptotic)
    whole = (b - a) * (fa + 4.0 * fc + fb) / 6.0

    top = 0
    stack_a[top] = a
    stack_b[top] = b
    stack_fa[top] = fa
    stack_fb[top] = fb
    stack_fc[top] = fc
    stack_whole[top] = whole
    stack_eps[top] = eps
    stack_depth[top] = max_depth
    top += 1

    result = 0.0
    while top > 0:
        top -= 1
        a = stack_a[top]
        b = stack_b[top]
        fa = stack_fa[top]
        fb = stack_fb[top]
        fc = stack_fc[top]
        whole = stack_whole[top]
        local_eps = stack_eps[top]
        depth = stack_depth[top]

        c = 0.5 * (a + b)
        left_mid = 0.5 * (a + c)
        right_mid = 0.5 * (c + b)
        fd = _z_eval_transformed(left_mid, q, k, df, log_const, s, asymptotic)
        fe = _z_eval_transformed(right_mid, q, k, df, log_const, s, asymptotic)
        left = (c - a) * (fa + 4.0 * fd + fc) / 6.0
        right = (b - c) * (fc + 4.0 * fe + fb) / 6.0
        delta = left + right - whole

        if depth <= 0 or abs(delta) <= 15.0 * local_eps or top + 2 >= max_stack:
            result += left + right + delta / 15.0
        else:
            half_eps = local_eps / 2.0
            stack_a[top] = c
            stack_b[top] = b
            stack_fa[top] = fc
            stack_fb[top] = fb
            stack_fc[top] = fe
            stack_whole[top] = right
            stack_eps[top] = half_eps
            stack_depth[top] = depth - 1
            top += 1

            stack_a[top] = a
            stack_b[top] = c
            stack_fa[top] = fa
            stack_fb[top] = fc
            stack_fc[top] = fd
            stack_whole[top] = left
            stack_eps[top] = half_eps
            stack_depth[top] = depth - 1
            top += 1

    return result


@njit(cache=True)
def _s_eval_transformed_iter(v, q, k, df, log_const, eps_z, max_depth_z):
    eps = 1e-12
    if v <= eps or v >= 1.0 - eps:
        return 0.0
    s = v / (1.0 - v)
    jac = 1.0 / ((1.0 - v) * (1.0 - v))
    return _adaptive_simpson_z_iter(q, k, df, log_const, s, False, eps_z, max_depth_z) * jac


@njit(cache=True)
def _adaptive_simpson_s_iter(q, k, df, log_const, eps_s, eps_z, max_depth_s, max_depth_z):
    max_stack = 4096
    stack_a = np.empty(max_stack, np.float64)
    stack_b = np.empty(max_stack, np.float64)
    stack_fa = np.empty(max_stack, np.float64)
    stack_fb = np.empty(max_stack, np.float64)
    stack_fc = np.empty(max_stack, np.float64)
    stack_whole = np.empty(max_stack, np.float64)
    stack_eps = np.empty(max_stack, np.float64)
    stack_depth = np.empty(max_stack, np.int64)

    a = 1e-10
    b = 1.0 - 1e-10
    c = 0.5 * (a + b)
    fa = _s_eval_transformed_iter(a, q, k, df, log_const, eps_z, max_depth_z)
    fb = _s_eval_transformed_iter(b, q, k, df, log_const, eps_z, max_depth_z)
    fc = _s_eval_transformed_iter(c, q, k, df, log_const, eps_z, max_depth_z)
    whole = (b - a) * (fa + 4.0 * fc + fb) / 6.0

    top = 0
    stack_a[top] = a
    stack_b[top] = b
    stack_fa[top] = fa
    stack_fb[top] = fb
    stack_fc[top] = fc
    stack_whole[top] = whole
    stack_eps[top] = eps_s
    stack_depth[top] = max_depth_s
    top += 1

    result = 0.0
    while top > 0:
        top -= 1
        a = stack_a[top]
        b = stack_b[top]
        fa = stack_fa[top]
        fb = stack_fb[top]
        fc = stack_fc[top]
        whole = stack_whole[top]
        local_eps = stack_eps[top]
        depth = stack_depth[top]

        c = 0.5 * (a + b)
        left_mid = 0.5 * (a + c)
        right_mid = 0.5 * (c + b)
        fd = _s_eval_transformed_iter(left_mid, q, k, df, log_const, eps_z, max_depth_z)
        fe = _s_eval_transformed_iter(right_mid, q, k, df, log_const, eps_z, max_depth_z)
        left = (c - a) * (fa + 4.0 * fd + fc) / 6.0
        right = (b - c) * (fc + 4.0 * fe + fb) / 6.0
        delta = left + right - whole

        if depth <= 0 or abs(delta) <= 15.0 * local_eps or top + 2 >= max_stack:
            result += left + right + delta / 15.0
        else:
            half_eps = local_eps / 2.0
            stack_a[top] = c
            stack_b[top] = b
            stack_fa[top] = fc
            stack_fb[top] = fb
            stack_fc[top] = fe
            stack_whole[top] = right
            stack_eps[top] = half_eps
            stack_depth[top] = depth - 1
            top += 1

            stack_a[top] = a
            stack_b[top] = c
            stack_fa[top] = fa
            stack_fb[top] = fc
            stack_fc[top] = fd
            stack_whole[top] = left
            stack_eps[top] = half_eps
            stack_depth[top] = depth - 1
            top += 1

    return result


@njit(cache=True)
def _studentized_range_sf_adaptive(q, k, df):
    if not (np.isfinite(q) and np.isfinite(k) and np.isfinite(df)):
        return np.nan
    if q < 0.0 or k <= 1.0 or df <= 0.0:
        return np.nan
    if df >= 100000.0:
        cdf = _adaptive_simpson_z_iter(q, k, df, 0.0, 1.0, True, 1e-9, 18)
    else:
        log_const = _studentized_log_cdf_const(k, df)
        cdf = _adaptive_simpson_s_iter(q, k, df, log_const, 1e-8, 1e-8, 14, 14)
    sf = 1.0 - cdf
    if sf < 0.0:
        return 0.0
    if sf > 1.0:
        return 1.0
    return sf


@njit(parallel=True, cache=True)
def _studentized_range_sf_adaptive_kernel(q_values, pair_k, pair_df):
    n_pairs, n_proteins = q_values.shape
    out = np.full((n_pairs, n_proteins), np.nan)
    for flat in prange(n_pairs * n_proteins):
        pair_idx = flat // n_proteins
        protein = flat - pair_idx * n_proteins
        out[pair_idx, protein] = _studentized_range_sf_adaptive(
            q_values[pair_idx, protein], pair_k[pair_idx, protein], pair_df[pair_idx, protein]
        )
    return out


@njit(cache=True)
def _ch_wprob(w, rr, cc, xleg, aleg):
    c1 = -30.0
    c2 = -50.0
    c3 = 60.0
    upper = 8.0
    qsqz = 0.5 * w

    if qsqz >= upper:
        return 1.0

    pr_w = 2.0 * _normal_cdf(qsqz) - 1.0
    if pr_w >= np.exp(c2 / cc):
        pr_w = pr_w ** cc
    else:
        pr_w = 0.0

    if w > 3.0:
        wincr = 2
    else:
        wincr = 3

    blb = qsqz
    binc = (upper - qsqz) / wincr
    bub = blb + binc
    einsum = 0.0
    cc1 = cc - 1.0

    for _ in range(wincr):
        elsum = 0.0
        center = 0.5 * (bub + blb)
        half_width = 0.5 * (bub - blb)

        for jj in range(12):
            if jj >= 6:
                j = 11 - jj
                xx = xleg[j]
            else:
                j = jj
                xx = -xleg[j]

            ac = center + half_width * xx
            qexpo = ac * ac
            if qexpo > c3:
                break

            pplus = 2.0 * _normal_cdf(ac)
            pminus = 2.0 * _normal_cdf(ac - w)
            rinsum = 0.5 * (pplus - pminus)
            if rinsum >= np.exp(c1 / cc1):
                elsum += aleg[j] * np.exp(-0.5 * qexpo) * rinsum ** cc1

        elsum *= (2.0 * half_width * cc) * INV_SQRT_2PI
        einsum += elsum
        blb = bub
        bub += binc

    pr_w += einsum
    if pr_w <= np.exp(c1 / rr):
        return 0.0
    pr_w = pr_w ** rr
    if pr_w >= 1.0:
        return 1.0
    return pr_w


@njit(cache=True)
def _studentized_range_cdf_ch(q, k, df, xleg, aleg, xlegq, alegq):
    rr = 1.0
    cc = k
    if not (np.isfinite(q) and np.isfinite(k) and np.isfinite(df)):
        return np.nan
    if q <= 0.0:
        return 0.0
    if df < 2.0 or rr < 1.0 or cc < 2.0:
        return np.nan
    if df > 25000.0:
        return _ch_wprob(q, rr, cc, xleg, aleg)

    f2 = 0.5 * df
    f2lf = (f2 * np.log(df)) - (df * np.log(2.0)) - math.lgamma(f2)
    f21 = f2 - 1.0
    ff4 = 0.25 * df

    if df <= 100.0:
        ulen = 1.0
    elif df <= 800.0:
        ulen = 0.5
    elif df <= 5000.0:
        ulen = 0.25
    else:
        ulen = 0.125

    f2lf += np.log(ulen)
    ans = 0.0
    last_otsum = 0.0

    for i in range(1, 51):
        otsum = 0.0
        twa1 = (2.0 * i - 1.0) * ulen

        for jj in range(16):
            if jj >= 8:
                j = jj - 8
                node = xlegq[j]
                t1 = (
                    f2lf
                    + f21 * np.log(twa1 + node * ulen)
                    - ((node * ulen + twa1) * ff4)
                )
                qsqz = q * np.sqrt((node * ulen + twa1) * 0.5)
            else:
                j = jj
                node = xlegq[j]
                t1 = (
                    f2lf
                    + f21 * np.log(twa1 - node * ulen)
                    + ((node * ulen - twa1) * ff4)
                )
                qsqz = q * np.sqrt((-node * ulen + twa1) * 0.5)

            if t1 >= -30.0:
                wprb = _ch_wprob(qsqz, rr, cc, xleg, aleg)
                otsum += wprb * alegq[j] * np.exp(t1)

        if i * ulen >= 1.0 and otsum <= 1e-14:
            break

        ans += otsum
        last_otsum = otsum

    if ans > 1.0:
        return 1.0
    return ans


@njit(cache=True)
def _studentized_range_sf_ch(q, k, df, xleg, aleg, xlegq, alegq):
    cdf = _studentized_range_cdf_ch(q, k, df, xleg, aleg, xlegq, alegq)
    if not np.isfinite(cdf):
        return np.nan
    sf = 1.0 - cdf
    if sf < 0.0:
        return 0.0
    if sf > 1.0:
        return 1.0
    return sf


@njit(parallel=True, cache=True)
def _studentized_range_sf_ch_kernel(q_values, pair_k, pair_df, xleg, aleg, xlegq, alegq):
    n_pairs, n_proteins = q_values.shape
    out = np.full((n_pairs, n_proteins), np.nan)
    for flat in prange(n_pairs * n_proteins):
        pair_idx = flat // n_proteins
        protein = flat - pair_idx * n_proteins
        out[pair_idx, protein] = _studentized_range_sf_ch(
            q_values[pair_idx, protein],
            pair_k[pair_idx, protein],
            pair_df[pair_idx, protein],
            xleg,
            aleg,
            xlegq,
            alegq,
        )
    return out


@njit(parallel=True, cache=True)
def _studentized_range_sf_kernel(q_values, pair_k, pair_df, s_nodes, s_weights, z_nodes, z_weights):
    n_pairs, n_proteins = q_values.shape
    out = np.full((n_pairs, n_proteins), np.nan)

    for flat in prange(n_pairs * n_proteins):
        pair_idx = flat // n_proteins
        protein = flat - pair_idx * n_proteins
        q = q_values[pair_idx, protein]
        df = pair_df[pair_idx, protein]
        kval = pair_k[pair_idx, protein]

        out[pair_idx, protein] = _studentized_range_sf(
            q, kval, df, s_nodes, s_weights, z_nodes, z_weights
        )

    return out


@njit(cache=True)
def _studentized_range_s_quadrature(df, s_nodes, s_weights):
    width = 10.0 / np.sqrt(2.0 * df)
    if width < 0.6:
        width = 0.6
    lo = 1.0 - width
    if lo < 1e-6:
        lo = 1e-6
    hi = 1.0 + width
    s_mid = 0.5 * (hi + lo)
    s_scale = 0.5 * (hi - lo)
    log_const = (
        np.log(2.0)
        + (df / 2.0) * np.log(df / 2.0)
        - math.lgamma(df / 2.0)
    )

    s_values = np.empty(s_nodes.size, np.float64)
    s_density_weights = np.empty(s_nodes.size, np.float64)
    for i in range(s_nodes.size):
        s = s_mid + s_scale * s_nodes[i]
        s_values[i] = s
        log_density = log_const + (df - 1.0) * np.log(s) - (df * s * s / 2.0)
        if log_density < -745.0:
            s_density_weights[i] = 0.0
        else:
            s_density_weights[i] = s_scale * s_weights[i] * np.exp(log_density)

    return s_values, s_density_weights


@njit(parallel=True, cache=True)
def _studentized_range_sf_grouped_kernel(
    flat_q,
    group_ids,
    group_k,
    group_df,
    group_s_values,
    group_s_density_weights,
    z_nodes,
    z_weights,
):
    out = np.full(flat_q.size, np.nan)
    zmax = 8.0

    for idx in prange(flat_q.size):
        group = group_ids[idx]
        if group < 0:
            continue

        q = flat_q[idx]
        k = group_k[group]
        df = group_df[group]
        if not np.isfinite(q) or q < 0.0 or k <= 1.0 or df <= 0.0:
            continue

        if df > 1000.0:
            out[idx] = _studentized_range_sf_asymptotic(q, k, z_nodes, z_weights)
            continue

        cdf = 0.0
        for si in range(group_s_values.shape[1]):
            s_weight_density = group_s_density_weights[group, si]
            if s_weight_density == 0.0:
                continue
            s = group_s_values[group, si]

            inner = 0.0
            for zi in range(z_nodes.size):
                z = zmax * z_nodes[zi]
                z_weight = zmax * z_weights[zi]
                phi = np.exp(-0.5 * z * z) * INV_SQRT_2PI
                delta = _normal_cdf(z + q * s) - _normal_cdf(z)
                if delta < 0.0:
                    delta = 0.0
                inner += z_weight * phi * delta ** (k - 1.0)

            cdf += s_weight_density * k * inner

        sf = 1.0 - cdf
        if sf < 0.0:
            out[idx] = 0.0
        elif sf > 1.0:
            out[idx] = 1.0
        else:
            out[idx] = sf

    return out


@njit(parallel=True, cache=True)
def _studentized_range_r_grid(k, r_max, grid_size, z_nodes, z_weights):
    r_values = np.linspace(0.0, r_max, grid_size)
    cdf_values = np.empty(grid_size, np.float64)
    zmax = 8.0

    for ri in prange(grid_size):
        r = r_values[ri]
        cdf = 0.0
        for zi in range(z_nodes.size):
            z = zmax * z_nodes[zi]
            z_weight = zmax * z_weights[zi]
            phi = np.exp(-0.5 * z * z) * INV_SQRT_2PI
            delta = _normal_cdf(z + r) - _normal_cdf(z)
            if delta < 0.0:
                delta = 0.0
            cdf += z_weight * k * phi * delta ** (k - 1.0)

        if cdf < 0.0:
            cdf_values[ri] = 0.0
        elif cdf > 1.0:
            cdf_values[ri] = 1.0
        else:
            cdf_values[ri] = cdf

    return r_values, cdf_values


@njit(cache=True)
def _interp_studentized_r_cdf(r, r_values, cdf_values):
    if r <= 0.0:
        return 0.0
    last = r_values.size - 1
    if r >= r_values[last]:
        return 1.0

    step = r_values[last] / last
    pos = r / step
    left = int(pos)
    if left >= last:
        return 1.0
    frac = pos - left
    return cdf_values[left] + frac * (cdf_values[left + 1] - cdf_values[left])


@njit(parallel=True, cache=True)
def _studentized_range_sf_grouped_interp_kernel(
    flat_q,
    group_ids,
    group_df,
    group_k_grid_ids,
    group_s_values,
    group_s_density_weights,
    k_r_values,
    k_r_cdf_values,
):
    out = np.full(flat_q.size, np.nan)

    for idx in prange(flat_q.size):
        group = group_ids[idx]
        if group < 0:
            continue

        q = flat_q[idx]
        df = group_df[group]
        if not np.isfinite(q) or q < 0.0 or df <= 0.0:
            continue

        k_grid = group_k_grid_ids[group]
        if df > 1000.0:
            cdf = _interp_studentized_r_cdf(
                q, k_r_values[k_grid], k_r_cdf_values[k_grid]
            )
        else:
            cdf = 0.0
            for si in range(group_s_values.shape[1]):
                cdf += group_s_density_weights[group, si] * _interp_studentized_r_cdf(
                    q * group_s_values[group, si],
                    k_r_values[k_grid],
                    k_r_cdf_values[k_grid],
                )

        sf = 1.0 - cdf
        if sf < 0.0:
            out[idx] = 0.0
        elif sf > 1.0:
            out[idx] = 1.0
        else:
            out[idx] = sf

    return out


def _studentized_range_sf_grouped_numba(q_values, pair_k, pair_df):
    flat_q = q_values.ravel()
    flat_k = pair_k.ravel()
    flat_df = pair_df.ravel()
    valid = np.isfinite(flat_q) & np.isfinite(flat_k) & np.isfinite(flat_df)
    if not valid.any():
        return np.full(q_values.shape, np.nan, dtype=np.float64)

    valid_idx = np.flatnonzero(valid)
    unique_groups, inverse = np.unique(
        np.column_stack((flat_k[valid], flat_df[valid])), axis=0, return_inverse=True
    )
    group_ids = np.full(flat_q.shape, -1, dtype=np.int64)
    group_ids[valid_idx] = inverse.astype(np.int64, copy=False)

    group_k = unique_groups[:, 0].astype(np.float64, copy=False)
    group_df = unique_groups[:, 1].astype(np.float64, copy=False)
    group_s_values = np.zeros((unique_groups.shape[0], S_NODES.size), dtype=np.float64)
    group_s_density_weights = np.zeros_like(group_s_values)
    for group, df in enumerate(group_df):
        if df <= 1000.0:
            s_values, s_density_weights = _studentized_range_s_quadrature(
                df, S_NODES, S_WEIGHTS
            )
            group_s_values[group] = s_values
            group_s_density_weights[group] = s_density_weights

    out = _studentized_range_sf_grouped_kernel(
        flat_q,
        group_ids,
        group_k,
        group_df,
        group_s_values,
        group_s_density_weights,
        Z_NODES,
        Z_WEIGHTS,
    )
    return out.reshape(q_values.shape)


def _studentized_range_sf_grouped_interp(q_values, pair_k, pair_df):
    flat_q = q_values.ravel()
    flat_k = pair_k.ravel()
    flat_df = pair_df.ravel()
    valid = np.isfinite(flat_q) & np.isfinite(flat_k) & np.isfinite(flat_df)
    if not valid.any():
        return np.full(q_values.shape, np.nan, dtype=np.float64)

    valid_idx = np.flatnonzero(valid)
    unique_groups, inverse = np.unique(
        np.column_stack((flat_k[valid], flat_df[valid])), axis=0, return_inverse=True
    )
    group_ids = np.full(flat_q.shape, -1, dtype=np.int64)
    group_ids[valid_idx] = inverse.astype(np.int64, copy=False)

    group_k = unique_groups[:, 0].astype(np.float64, copy=False)
    group_df = unique_groups[:, 1].astype(np.float64, copy=False)
    group_s_values = np.zeros((unique_groups.shape[0], S_NODES.size), dtype=np.float64)
    group_s_density_weights = np.zeros_like(group_s_values)
    unique_k, group_k_grid_ids = np.unique(group_k, return_inverse=True)
    group_k_grid_ids = group_k_grid_ids.astype(np.int64, copy=False)
    k_r_values = np.zeros((unique_k.size, STUDENTIZED_R_GRID_SIZE), dtype=np.float64)
    k_r_cdf_values = np.zeros_like(k_r_values)

    for k_idx, kval in enumerate(unique_k):
        r_values, r_cdf_values = _studentized_range_r_grid(
            kval, STUDENTIZED_R_GRID_MAX, STUDENTIZED_R_GRID_SIZE, Z_NODES, Z_WEIGHTS
        )
        k_r_values[k_idx] = r_values
        k_r_cdf_values[k_idx] = r_cdf_values

    for group, df in enumerate(group_df):
        if df <= 1000.0:
            s_values, s_density_weights = _studentized_range_s_quadrature(
                df, S_NODES, S_WEIGHTS
            )
            group_s_values[group] = s_values
            group_s_density_weights[group] = s_density_weights

    out = _studentized_range_sf_grouped_interp_kernel(
        flat_q,
        group_ids,
        group_df,
        group_k_grid_ids,
        group_s_values,
        group_s_density_weights,
        k_r_values,
        k_r_cdf_values,
    )
    return out.reshape(q_values.shape)


def _sorted_codes(values):
    uniques = np.array(sorted(pd.unique(values)), dtype=object)
    codes = pd.Categorical(values, categories=uniques, ordered=True).codes.astype(np.int64)
    return uniques, codes


def _first_seen_codes(values):
    uniques = np.array(pd.unique(values), dtype=object)
    codes = pd.Categorical(values, categories=uniques, ordered=True).codes.astype(np.int64)
    return uniques, codes


def _build_pairs(n_conditions):
    return np.array(
        [(a, b) for a in range(n_conditions) for b in range(a + 1, n_conditions)],
        dtype=np.int64,
    )


def _make_anova_frame(
    proteins,
    presence_mask,
    n_total,
    k,
    df_between,
    df_within,
    f_value,
    p_value,
    q_value,
    p_cut,
    fdr,
):
    anova = pd.DataFrame(
        {
            "Protein.Group": proteins,
            "group_presence_mask": presence_mask,
            "n_total": n_total,
            "n_groups_present": k,
            "df_between": df_between,
            "df_within": df_within,
            "F-value ": f_value,
            "p-value ": p_value,
            "anova_q_value": q_value,
            "anova_q_value_global": q_value,
        }
    )
    anova["adjusted p-value "] = anova["anova_q_value"]
    significant = (
        np.isfinite(p_value)
        & np.isfinite(q_value)
        & (p_value <= p_cut)
        & (q_value <= fdr)
    )
    anova["significant_bool"] = significant
    anova["significant"] = np.where(
        significant,
        anova["p-value "].astype(object),
        "ns",
    )
    return anova


def _tukey_fdr_arrays(tukey_p, anova_significant_mask):
    gated_p = mask_tukey_by_protein(tukey_p, anova_significant_mask)
    return {
        "global": bh_adjust_flat_matrix(tukey_p),
        "by_comparison": bh_adjust_by_comparison(tukey_p),
        "by_protein": bh_adjust_by_protein(tukey_p),
        "gated_global": bh_adjust_flat_matrix(gated_p),
        "gated_by_comparison": bh_adjust_by_comparison(gated_p),
        "gated_by_protein": bh_adjust_by_protein(gated_p),
    }


def _make_tukey_long_frame(
    proteins,
    presence_mask,
    conditions,
    pairs,
    tukey_stat,
    tukey_p,
    tukey_q,
    p_cut,
    fdr,
):
    frames = []
    for pair_idx, (a, b) in enumerate(pairs):
        p_values = tukey_p[pair_idx]
        frame = pd.DataFrame(
            {
                "Protein.Group": proteins,
                "group_presence_mask": presence_mask,
                "group_a": conditions[a],
                "group_b": conditions[b],
                "comparison": f"{conditions[a]}-{conditions[b]}",
                "tukey_statistic": tukey_stat[pair_idx],
                "tukey_p_value": p_values,
            }
        )
        frame["significant"] = frame["tukey_p_value"] <= p_cut
        for scope, q_values in tukey_q.items():
            column = f"tukey_q_value_{scope}"
            frame[column] = q_values[pair_idx]
            frame[f"significant_{scope}"] = frame[column] <= fdr
        frames.append(frame)

    if not frames:
        columns = [
            "Protein.Group",
            "group_presence_mask",
            "group_a",
            "group_b",
            "comparison",
            "tukey_statistic",
            "tukey_p_value",
            "significant",
        ]
        for scope in tukey_q:
            columns.extend([f"tukey_q_value_{scope}", f"significant_{scope}"])
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _scipy_studentized_range_sf_vector(q_values, pair_k, pair_df):
    flat_q = q_values.ravel()
    flat_k = pair_k.ravel()
    flat_df = pair_df.ravel()
    out = np.full(flat_q.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flat_q) & np.isfinite(flat_k) & np.isfinite(flat_df)
    out[valid] = stats.studentized_range.sf(flat_q[valid], flat_k[valid], flat_df[valid])
    return out.reshape(q_values.shape)


def _scipy_studentized_range_sf_grouped(q_values, pair_k, pair_df):
    flat_q = q_values.ravel()
    flat_k = pair_k.ravel()
    flat_df = pair_df.ravel()
    out = np.full(flat_q.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flat_q) & np.isfinite(flat_k) & np.isfinite(flat_df)
    if not valid.any():
        return out.reshape(q_values.shape)

    # q is usually unique, but k/df repeat enough to let SciPy receive q batches.
    keys = pd.DataFrame(
        {
            "idx": np.flatnonzero(valid),
            "k": flat_k[valid],
            "df": flat_df[valid],
        }
    )
    for (_, _), group in keys.groupby(["k", "df"], sort=False):
        idx = group["idx"].to_numpy(np.int64)
        out[idx] = stats.studentized_range.sf(flat_q[idx], flat_k[idx[0]], flat_df[idx[0]])

    return out.reshape(q_values.shape)


def anova_test(
    res_wide,
    p_cut=0.01,
    fdr=0.01,
    tukey_sf_backend="numba",
    verbose=True,
    return_tukey_long=False,
):
    timings = []
    t_total = perf_counter()

    def checkpoint(name, start):
        timings.append((name, perf_counter() - start))
        return perf_counter()

    if verbose:
        print("ANOVA, comparing all conditions now.\n")

    t = perf_counter()
    proteins, protein_codes = _sorted_codes(res_wide["Protein.Group"])
    conditions, condition_codes = _first_seen_codes(res_wide["condition"])
    intensity = res_wide["Intensity"].to_numpy(np.float64)
    t = checkpoint("prepare codes", t)

    counts, sums, sums_sq = _accumulate_summaries(
        protein_codes, condition_codes, intensity, proteins.size, conditions.size
    )
    t = checkpoint("numba accumulate summaries", t)

    presence_mask = presence_masks_from_counts(counts)
    t = checkpoint("numba presence masks", t)

    n_total, k, ssb, ssw, df_between, df_within, msw, f_value = _anova_kernel(
        counts, sums, sums_sq
    )
    t = checkpoint("numba anova kernel", t)

    p_value = np.full_like(f_value, np.nan)
    valid_f = np.isfinite(f_value)
    p_value[valid_f] = stats.f.sf(f_value[valid_f], df_between[valid_f], df_within[valid_f])
    anova_q_value = bh_adjust_1d(p_value)
    anova_results = _make_anova_frame(
        proteins,
        presence_mask,
        n_total,
        k,
        df_between,
        df_within,
        f_value,
        p_value,
        anova_q_value,
        p_cut,
        fdr,
    )
    t = checkpoint("anova p-values/results", t)

    pairs = _build_pairs(conditions.size)
    q_values, pair_k, pair_df = _tukey_q_kernel(counts, sums, msw, k, df_within, pairs)
    t = checkpoint("numba tukey q kernel", t)

    if tukey_sf_backend == "numba":
        tukey_p = _studentized_range_sf_kernel(
            q_values, pair_k, pair_df, S_NODES, S_WEIGHTS, Z_NODES, Z_WEIGHTS
        )
        t = checkpoint("numba studentized_range.sf", t)
    elif tukey_sf_backend == "numba_adaptive":
        tukey_p = _studentized_range_sf_adaptive_kernel(q_values, pair_k, pair_df)
        t = checkpoint("numba adaptive studentized_range.sf", t)
    elif tukey_sf_backend == "numba_ch":
        tukey_p = _studentized_range_sf_ch_kernel(
            q_values, pair_k, pair_df, CH_XLEG, CH_ALEG, CH_XLEGQ, CH_ALEGQ
        )
        t = checkpoint("numba Copenhaver-Holland studentized_range.sf", t)
    elif tukey_sf_backend == "numba_grouped":
        tukey_p = _studentized_range_sf_grouped_numba(q_values, pair_k, pair_df)
        t = checkpoint("numba grouped studentized_range.sf", t)
    elif tukey_sf_backend == "numba_grouped_interp":
        tukey_p = _studentized_range_sf_grouped_interp(q_values, pair_k, pair_df)
        t = checkpoint("numba grouped interpolated studentized_range.sf", t)
    elif tukey_sf_backend == "scipy_vector":
        tukey_p = _scipy_studentized_range_sf_vector(q_values, pair_k, pair_df)
        t = checkpoint("scipy vector studentized_range.sf", t)
    elif tukey_sf_backend == "scipy_grouped":
        tukey_p = _scipy_studentized_range_sf_grouped(q_values, pair_k, pair_df)
        t = checkpoint("scipy grouped studentized_range.sf", t)
    else:
        raise ValueError(
            "tukey_sf_backend must be one of: "
            "'numba', 'numba_adaptive', 'numba_ch', 'numba_grouped', "
            "'numba_grouped_interp', "
            "'scipy_vector', 'scipy_grouped'"
        )

    anova_significant_mask = (
        np.isfinite(anova_q_value) & (anova_q_value <= fdr) & (p_value <= p_cut)
    )
    tukey_q = _tukey_fdr_arrays(tukey_p, anova_significant_mask)
    t = checkpoint("tukey fdr corrections", t)

    tukey_results = pd.DataFrame(index=proteins)
    for pair_idx, (a, b) in enumerate(pairs):
        tukey_results[f"{conditions[a]}-{conditions[b]}"] = tukey_p[pair_idx]
    tukey_results.index.name = "Protein.Group"
    t = checkpoint("tukey dataframe", t)

    if return_tukey_long:
        tukey_long = _make_tukey_long_frame(
            proteins,
            presence_mask,
            conditions,
            pairs,
            q_values,
            tukey_p,
            tukey_q,
            p_cut,
            fdr,
        )
        t = checkpoint("tukey long dataframe", t)

    timings.append(("total", perf_counter() - t_total))
    if verbose:
        print("Perf timings:")
        for name, elapsed in sorted(timings, key=lambda item: item[1], reverse=True):
            print(f"{elapsed:9.4f}s  {name}")

    if return_tukey_long:
        return anova_results, tukey_results, tukey_long, timings
    return anova_results, tukey_results, timings
