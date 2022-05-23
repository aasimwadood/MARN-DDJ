import numpy as np
from scipy.ndimage.filters import convolve
from numba import jit
def postprocessing_step(RGB_d, eps=0.01):
    R_d, G_d, B_d = tsplit(RGB_d)
    img_shape = RGB_d.shape[:2]
    r_mask, g_mask, b_mask = masks_CFA_Bayer(img_shape)
    post_diff_gr = G_d - R_d
    post_diff_gb = G_d - B_d

    post_h5 = np.array([-5, 15, 44, 15, -5]) / 64.

    post_P_mat = to_convolve_mat(np.eye(5) * post_h5)
    post_Q_mat = to_convolve_mat(np.flip(np.eye(5), 1) * post_h5)
    h_mat = np.vstack([np.zeros((2, 5)), post_h5.reshape((1, 5)), np.zeros((2, 5))])
    post_H_mat = to_convolve_mat(h_mat)
    post_V_mat = to_convolve_mat(np.transpose(h_mat))

    def process_color(post_diff_color):
        D_p = convolve(post_diff_color, post_P_mat)
        D_q = convolve(post_diff_color, post_Q_mat)
        D_h = convolve(post_diff_color, post_H_mat)
        D_v = convolve(post_diff_color, post_V_mat)

        lambda_p = gradient_p(post_diff_color)
        lambda_q = gradient_q(post_diff_color)
        lambda_h = gradient_h(post_diff_color)
        lambda_v = gradient_v(post_diff_color)

        lambdas = [lambda_p, lambda_q, lambda_h, lambda_v]
        etas = np.asarray(map(lambda x: 1. / (x + eps), lambdas))

        etas_sum = np.apply_along_axis(np.sum, 0, etas)

        post_est_sum = np.apply_along_axis(np.sum, 0, etas * np.asarray([D_p, D_q, D_h, D_v]))
        post_diff_p = post_est_sum / etas_sum

        post_diff_p[:2, :] = post_diff_color[:2, :]
        post_diff_p[-2:, :] = post_diff_color[-2:, :]
        post_diff_p[:, :2] = post_diff_color[:, :2]
        post_diff_p[:, -2:] = post_diff_color[:, -2:]

        return post_diff_p

    post_diff_gr_p = process_color(post_diff_gr)
    post_diff_gb_p = process_color(post_diff_gb)

    rb_mask = np.asarray(1 - g_mask, dtype=bool)
    rg_mask = np.asarray(1 - b_mask, dtype=bool)
    bg_mask = np.asarray(1 - r_mask, dtype=bool)

    G_refined = (0.5 * (R_d + post_diff_gr_p) + 0.5 * (B_d + post_diff_gb_p)) * rb_mask + G_d * g_mask
    assert (np.sum(np.isnan(G_refined)) == 0)

    post_diff_rg = R_d - G_refined
    post_diff_bg = B_d - G_refined

    post_diff_rg_p = process_color(post_diff_rg)
    post_diff_bg_p = process_color(post_diff_bg)

    R_refined = (G_refined + post_diff_rg_p) * bg_mask + R_d * r_mask
    B_refined = (G_refined + post_diff_bg_p) * rg_mask + B_d * b_mask

    assert (np.sum(np.isnan(R_refined)) == 0)
    assert (np.sum(np.isnan(B_refined)) == 0)

    return norm_img(tstack([R_refined, G_refined, B_refined]))

def tstack(a):
    a = np.asarray(a)
    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)

## Norm color range in images
def norm_color(color):
    color = np.where(color < 0, 0., color)
    color = np.where(color > 255, 255., color)
    return color


def norm_img(img):
    R, G, B = tsplit(img)
    R = norm_color(R)
    G = norm_color(G)
    B = norm_color(B)
    return tstack([R, G, B])


def tsplit(a):
    a = np.asarray(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])

def masks_CFA_Bayer(shape):
    pattern = 'RGGB'
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')

def to_convolve_mat(mat):
    dims = len(mat.shape)
    res = np.copy(mat)
    for i in range(dims):
        res = np.flip(res, i)
    return res


@jit
def gradient_p(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in range(1, shape[0] - 1, 1):
        for j in range(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i - 1][j - 1] - ar[i + 1][j + 1]) + np.abs(ar[i - 1][j - 1] - ar[i][j]) + \
                           np.abs(ar[i + 1][j + 1] - ar[i][j])
    return result


@jit
def gradient_q(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in range(1, shape[0] - 1, 1):
        for j in range(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i - 1][j + 1] - ar[i + 1][j - 1]) + np.abs(ar[i - 1][j + 1] - ar[i][j]) + \
                           np.abs(ar[i + 1][j - 1] - ar[i][j])
    return result


@jit
def gradient_h(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in range(0, shape[0], 1):
        for j in range(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i][j - 1] - ar[i][j + 1]) + np.abs(ar[i][j - 1] - ar[i][j]) + \
                           np.abs(ar[i][j + 1] - ar[i][j])
    return result


@jit
def gradient_v(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in range(1, shape[0] - 1, 1):
        for j in range(0, shape[1], 1):
            result[i][j] = np.abs(ar[i - 1][j] - ar[i + 1][j]) + np.abs(ar[i - 1][j] - ar[i][j]) + \
                           np.abs(ar[i + 1][j] - ar[i][j])
    return result
