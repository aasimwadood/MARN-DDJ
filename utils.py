import math
import torch
import torch as th
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
# from skimage.measure import structural_similarity
from skimage import measure
from PIL import Image
import cv2
from torch.autograd import Variable


def tensor2Im(img, dtype=np.float32):
    assert (isinstance(img, th.Tensor) and img.ndimension() == 4), "A 4D " \
                                                                   "torch.Tensor is expected."
    fshape = (0, 2, 3, 1)

    return img.detach().numpy().transpose(fshape).astype(dtype)


def im2Tensor(img, dtype=th.FloatTensor):
    assert (isinstance(img, np.ndarray) and img.ndim in (2, 3, 4)), "A numpy " \
                                                                    "nd array of dimensions 2, 3, or 4 is expected."

    if img.ndim == 2:
        return th.from_numpy(img).unsqueeze_(0).unsqueeze_(0).type(dtype)
    elif img.ndim == 3:
        return th.from_numpy(img.transpose(2, 0, 1)).unsqueeze_(0).type(dtype)
    else:
        return th.from_numpy(img.transpose((3, 2, 0, 1))).type(dtype)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def weigths_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


import torchvision


def unnormalize(images):
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def calculate_psnr_fast(target, prediction):
    # Calculate PSNR
    # Data have to be in range (0, 1)
    assert prediction.max().cpu().data.numpy() <= 1
    assert prediction.min().cpu().data.numpy() >= 0
    psnr_list = []
    # print(prediction.size(0))
    for i in range(prediction.size(0)):
        mse = torch.mean(torch.pow(prediction.data[i] - target.data[i], 2))
        try:
            psnr_list.append(10 * np.log10(1 ** 2 / mse))
        except:
            print('error in psnr calculation')
            continue
    return psnr_list


# transforms RGB image to gray and returns image of the form H,W
def to_gray(im):
    if im.shape[0] == 3:
        im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    return im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140


# compute the structured similarity index
def ssim(original, result, max_intensity, crop=7):
    # color image in the form 3HW or 4HW
    original = original[0:3, crop:-crop, crop:-crop]
    result = result[0:3, crop:-crop, crop:-crop]
    original = to_gray(original)
    result = to_gray(result)

    original = np.maximum(0, np.minimum(original, max_intensity))
    result = np.maximum(0, np.minimum(result, max_intensity))

    return measure.compare_ssim(original.astype("float32"),
                                 result.astype("float32"),
                                 dynamic_range=max_intensity)


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def remosaicv2(i_rgb, type):
    batch_size, ch, patch_height, patch_width = i_rgb.size()

    # bayer = torch.zeros(batch_size, 1, patch_height, patch_width).cuda()

    # bayer[:, 0, ::2, ::2] = i_rgb[:, 1, ::2, ::2]  # G
    # bayer[:, 0, ::2, 1::2] = i_rgb[:, 0, ::2, 1::2]  # R
    # bayer[:, 0, 1::2, ::2] = i_rgb[:, 2, 1::2, ::2]  # B
    # bayer[:, 0, 1::2, 1::2] = i_rgb[:, 1, 1::2, 1::2]  # G

    bayer_rgb = torch.zeros(batch_size, 3, patch_height, patch_width).cuda()

    bayer_rgb[:, 1, ::2, ::2] = i_rgb[:, 0, ::2, ::2]  # G
    bayer_rgb[:, 0, ::2, 1::2] = i_rgb[:, 0, ::2, 1::2]  # R
    bayer_rgb[:, 2, 1::2, ::2] = i_rgb[:, 0, 1::2, ::2]  # B
    bayer_rgb[:, 1, 1::2, 1::2] = i_rgb[:, 0, 1::2, 1::2]  # G

    if type == 0:
        o_bayer = bayer
    else:
        o_bayer = bayer_rgb
    return o_bayer


def remosaic(i_rgb, type):
    ch, patch_height, patch_width = i_rgb.size()

    bayer = torch.zeros(1, patch_height, patch_width)

    bayer[0, ::2, ::2] = i_rgb[1, ::2, ::2]  # G
    bayer[0, ::2, 1::2] = i_rgb[0, ::2, 1::2]  # R
    bayer[0, 1::2, ::2] = i_rgb[2, 1::2, ::2]  # B
    bayer[0, 1::2, 1::2] = i_rgb[1, 1::2, 1::2]  # G

    bayer_rgb = torch.zeros(3, patch_height, patch_width)

    bayer_rgb[1, ::2, ::2] = bayer[0, ::2, ::2]  # G
    bayer_rgb[0, ::2, 1::2] = bayer[0, ::2, 1::2]  # R
    bayer_rgb[2, 1::2, ::2] = bayer[0, 1::2, ::2]  # B
    bayer_rgb[1, 1::2, 1::2] = bayer[0, 1::2, 1::2]  # G

    if type == 0:
        o_bayer = bayer
    else:
        o_bayer = bayer_rgb

    return o_bayer


def demosaic_cv2(bayer, bseq):
    batch_size, ch, patch_height, patch_width = bayer.size()
    patch_size = patch_width

    bseq = cv2.COLOR_BAYER_GR2BGR

    bayer_np = bayer.numpy()
    # unnormalize
    bayer_np = ((bayer_np / 2 + 0.5) * 255)

    # to uint8
    bayer_u8 = bayer_np.astype('uint8')
    debayer = np.zeros([batch_size, patch_size, patch_size, 3])

    for i in range(batch_size):
        debayer[i] = cv2.cvtColor(bayer_u8[i, 0], cv2.COLOR_BAYER_GR2BGR)  # bilinear intp.
        # debayer[i] = cv2.demosaicing(bayer_u8[i,0], cv2.COLOR_BAYER_GR2BGR_EA)  # so so
        # debayer[i] = cv2.demosaicing(bayer_u8[i, 0], cv2.COLOR_BAYER_GR2BGR_VNG)  # bad

    # normalized
    debayer = debayer.astype('float32') / 255
    debayer = (debayer - 0.5) * 2

    # to torch.FloatTensor
    dem0 = torch.from_numpy(debayer)

    # b, h, w, ch =>  b, ch, h, w
    dem = torch.zeros(batch_size, 3, patch_size, patch_size)
    dem[:, 0, :, :] = dem0[:, :, :, 0]
    dem[:, 1, :, :] = dem0[:, :, :, 1]
    dem[:, 2, :, :] = dem0[:, :, :, 2]

    return dem


import numpy as np
from scipy import signal


def dem_gaussian(i_bayer):
    # GRBG case only
    ch, patch_height, patch_width = i_bayer.size()
    out = torch.zeros(3, patch_height, patch_width)
    img = i_bayer[0]
    x = img
    w_k = np.array([[1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1]],
                   dtype='float') / 256

    w0 = np.zeros((5, 5))
    w1 = np.zeros((5, 5))
    w2 = np.zeros((5, 5))
    w3 = np.zeros((5, 5))

    w0[::2, ::2] = w_k[::2, ::2]  # c9
    w1[::2, 1::2] = w_k[::2, 1::2]  # v6
    w2[1::2, ::2] = w_k[1::2, ::2]  # h6
    w3[1::2, 1::2] = w_k[1::2, 1::2]  # d4

    o0 = signal.convolve2d(x, w0, boundary='symm', mode='same')
    o1 = signal.convolve2d(x, w1, boundary='symm', mode='same')
    o2 = signal.convolve2d(x, w2, boundary='symm', mode='same')
    o3 = signal.convolve2d(x, w3, boundary='symm', mode='same')

    gr = np.zeros((patch_height, patch_width))
    r = np.zeros((patch_height, patch_width))
    b = np.zeros((patch_height, patch_width))
    gb = np.zeros((patch_height, patch_width))

    gr[::2, ::2] = o0[::2, ::2]
    gr[::2, 1::2] = o1[::2, 1::2]
    gr[1::2, ::2] = o2[1::2, ::2]
    gr[1::2, 1::2] = o3[1::2, 1::2]

    r[::2, ::2] = o1[::2, ::2]
    r[::2, 1::2] = o0[::2, 1::2]
    r[1::2, ::2] = o3[1::2, ::2]
    r[1::2, 1::2] = o2[1::2, 1::2]

    b[::2, ::2] = o2[::2, ::2]
    b[::2, 1::2] = o3[::2, 1::2]
    b[1::2, ::2] = o0[1::2, ::2]
    b[1::2, 1::2] = o1[1::2, 1::2]

    gb[::2, ::2] = o3[::2, ::2]
    gb[::2, 1::2] = o2[::2, 1::2]
    gb[1::2, ::2] = o1[1::2, ::2]
    gb[1::2, 1::2] = o0[1::2, 1::2]

    o_r = r * 4
    o_g = ((gr + gb) * 4) / 2
    o_b = b * 4

    o_dem = np.zeros((3, patch_height, patch_width))
    o_dem[0] = o_r
    o_dem[1] = o_g
    o_dem[2] = o_b

    out = torch.from_numpy(o_dem)

    # imshow(torchvision.utils.make_grid(out), 3)
    return out


def dem_gaussianv2(i_bayer):
    # GRBG case only
    batch_size, ch, patch_height, patch_width = i_bayer.size()
    out = torch.zeros(batch_size, 3, patch_height, patch_width).cuda()

    for i in range(batch_size):
        img = i_bayer[i, 0]
        x = img
        w_k = np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]],
                       dtype='float') / 256

        w0 = np.zeros((5, 5))
        w1 = np.zeros((5, 5))
        w2 = np.zeros((5, 5))
        w3 = np.zeros((5, 5))

        w0[::2, ::2] = w_k[::2, ::2]  # c9
        w1[::2, 1::2] = w_k[::2, 1::2]  # v6
        w2[1::2, ::2] = w_k[1::2, ::2]  # h6
        w3[1::2, 1::2] = w_k[1::2, 1::2]  # d4

        o0 = signal.convolve2d(x, w0, boundary='symm', mode='same')
        o1 = signal.convolve2d(x, w1, boundary='symm', mode='same')
        o2 = signal.convolve2d(x, w2, boundary='symm', mode='same')
        o3 = signal.convolve2d(x, w3, boundary='symm', mode='same')

        gr = np.zeros((patch_height, patch_width))
        r = np.zeros((patch_height, patch_width))
        b = np.zeros((patch_height, patch_width))
        gb = np.zeros((patch_height, patch_width))

        gr[::2, ::2] = o0[::2, ::2]
        gr[::2, 1::2] = o1[::2, 1::2]
        gr[1::2, ::2] = o2[1::2, ::2]
        gr[1::2, 1::2] = o3[1::2, 1::2]

        r[::2, ::2] = o1[::2, ::2]
        r[::2, 1::2] = o0[::2, 1::2]
        r[1::2, ::2] = o3[1::2, ::2]
        r[1::2, 1::2] = o2[1::2, 1::2]

        b[::2, ::2] = o2[::2, ::2]
        b[::2, 1::2] = o3[::2, 1::2]
        b[1::2, ::2] = o0[1::2, ::2]
        b[1::2, 1::2] = o1[1::2, 1::2]

        gb[::2, ::2] = o3[::2, ::2]
        gb[::2, 1::2] = o2[::2, 1::2]
        gb[1::2, ::2] = o1[1::2, ::2]
        gb[1::2, 1::2] = o0[1::2, 1::2]

        o_r = r * 4
        o_g = ((gr + gb) * 4) / 2
        o_b = b * 4

        o_dem = np.zeros((3, patch_height, patch_width))
        o_dem[0] = o_r
        o_dem[1] = o_g
        o_dem[2] = o_b

        out = torch.from_numpy(o_dem)

    # imshow(torchvision.utils.make_grid(out), 3)
    print(out)
    return out

def init_colortransformation_gamma():
    gammaparams = np.load('gammaparams.npy').astype('float32')

    colortrans_mtx = np.load('colortrans.npy').astype('float32')
    colortrans_mtx = np.expand_dims(np.expand_dims(colortrans_mtx, 0), 0)

    param_dict = {
        'UINT8': 255.0,
        'UINT16': 65535.0,
        'corr_const': 15.0,
        'gammaparams': gammaparams,
        'colortrans_mtx': colortrans_mtx,
    }

    return param_dict

    # compute the gamma function


# we fitted a function according to the given gamma mapping in the
# Microsoft demosaicing data set
def _f_gamma(img, param_dict):
    params = param_dict['gammaparams']
    UINT8 = param_dict['UINT8']
    UINT16 = param_dict['UINT16']

    return UINT8 * (((1 + params[0]) * \
                     np.power(UINT16 * (img / UINT8), 1.0 / params[1]) - \
                     params[0] +
                     params[2] * (UINT16 * (img / UINT8))) / UINT16)


# apply the color transformation matrix
def _f_color_t(img, param_dict):
    return np.tensordot(param_dict['colortrans_mtx'], img, axes=([1, 2], [0, 1]))


# apply the black level correction constant
def _f_corr(img, param_dict):
    return img - param_dict['UINT8'] * \
           (param_dict['corr_const'] / param_dict['UINT16'])

#wrapper for the conversion from linear to sRGB space with given parameters
#def apply_colortransformation_gamma(img, param_dict):
#    img = _f_color_t(img, param_dict)
#    img = np.where( img > 0.0, _f_gamma(img, param_dict), img )
#    img = _f_corr(img, param_dict)

#    return img
# wrapper for the conversion from linear to sRGB space with given parameters
def apply_colortransformation_gamma(img, param_dict):
    # assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
    img = _f_color_t(img, param_dict)
    img = np.where(img > 0.0, _f_gamma(img, param_dict), img)
    img = _f_corr(img, param_dict)

    return img

def calculate_psnr_fast_srgb(prediction, target):
    avg_psnr = 0

    srgb_params = init_colortransformation_gamma()
    psnr_list = []
    result_rgb = apply_colortransformation_gamma(np.expand_dims(prediction,0), srgb_params)
    target_rgb = apply_colortransformation_gamma(np.expand_dims(target,0), srgb_params)
    return calculate_psnr_fast(target_rgb[0]/255, result_rgb[0]/255), torch.FloatTensor(result_rgb[None, :])

def im2Tensor(img, dtype=th.FloatTensor):
    assert (isinstance(img, np.ndarray) and img.ndim in (2, 3, 4)), "A numpy " \
                                                                        "nd array of dimensions 2, 3, or 4 is expected."

    if img.ndim == 2:
        return th.from_numpy(img).unsqueeze_(0).unsqueeze_(0).type(dtype)
    elif img.ndim == 3:
        return th.from_numpy(img.transpose(2, 0, 1)).unsqueeze_(0).type(dtype)
    else:
        return th.from_numpy(img.transpose((3, 2, 0, 1))).type(dtype)

#compute the gamma function
#we fitted a function according to the given gamma mapping in the
#Microsoft demosaicing data set
def _f_gamma(img, param_dict):
    params = param_dict['gammaparams']
    UINT8 = param_dict['UINT8']
    UINT16 = param_dict['UINT16']

    return UINT8*(((1 + params[0]) * \
        np.power(UINT16*(img/UINT8), 1.0/params[1]) - \
        params[0] +
        params[2]*(UINT16*(img/UINT8)))/UINT16)

#apply the color transformation matrix
def _f_color_t(img, param_dict):
    return  np.tensordot(param_dict['colortrans_mtx'], img, axes=([1,2],[0,1]))

#apply the black level correction constant
def _f_corr(img, param_dict):
    return img - param_dict['UINT8'] * \
         (param_dict['corr_const']/param_dict['UINT16'])



def tensor2Im(img, dtype=np.float32):
    assert (isinstance(img, th.Tensor) and img.ndimension() == 4), "A 4D " \
                                                                   "torch.Tensor is expected."
    fshape = (0, 2, 3, 1)

    return img.numpy().transpose(fshape).astype(dtype)
