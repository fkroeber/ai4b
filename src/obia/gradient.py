"""
Edge detection code snippets based on the skimage implementation of canny,
Functions to return intermediate processing results of canny (such as the gradient)
and to post-process a given gradient-map using canny's utilities (such as NMS & hysteresis thresholding). 
"""

import numpy as np
import scipy.ndimage as ndi

from skimage.util.dtype import dtype_limits
from skimage._shared.utils import check_nD
from skimage.feature._canny import _preprocess, _get_local_maxima
from scipy.ndimage import filters


def canny_grad(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=True,
    *,
    mode="constant",
    cval=0.0
):
    """
    Edge filter an image using canny algorithm.
    Returns gradient at edge locations instead of binarised edge image.
    """

    # checks and threshold setting
    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    # image filtering
    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode, cval)

    # gradient magnitude estimation
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = np.hypot(isobel, jsobel)

    if use_quantiles:
        low_threshold, high_threshold = np.percentile(
            magnitude, [100.0 * low_threshold, 100.0 * high_threshold]
        )

    # non-maximum suppression
    local_maxima = _get_local_maxima(isobel, jsobel, magnitude, eroded_mask)

    # double thresholding and edge traking
    low_mask = local_maxima & (magnitude >= low_threshold)

    # segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = local_maxima & (magnitude >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]

    # return edge gradient
    grad = np.where(output_mask, magnitude, 0)
    return grad


def sobel_grad(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=True,
    *,
    mode="constant",
    cval=0.0
):
    """
    Smooth an image & return sobel gradients.
    """

    # Image filtering
    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode, cval)

    # Gradient magnitude estimation
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = np.hypot(isobel, jsobel)

    return isobel, jsobel, magnitude, eroded_mask


def canny_postprocess(
    isobel,
    jsobel,
    magnitude,
    eroded_mask,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=True,
    *,
    mode="constant",
    cval=0.0
):
    """
    Post-processing of an input using the later parts of canny's algorithm.
    Perform NMS & hystersis edge tracking based on input maps for edge intensities.
    """

    # checks and threshold setting
    check_nD(magnitude, 2)
    dtype_max = dtype_limits(magnitude, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    if use_quantiles:
        low_threshold, high_threshold = np.nanpercentile(
            magnitude, [100.0 * low_threshold, 100.0 * high_threshold]
        )

    # non-maximum suppression
    local_maxima = _get_local_maxima(isobel, jsobel, magnitude, eroded_mask)

    # double thresholding and edge tracking
    low_mask = local_maxima & (magnitude >= low_threshold)

    # segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = local_maxima & (magnitude >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]

    # return edge gradient
    grad = np.where(output_mask, magnitude, 0)
    return grad


# sobel gradient
def gradient(image):
    Ix = filters.sobel(image, axis=1)
    Iy = filters.sobel(image, axis=0)
    return Ix, Iy


# src: Van De Weijer et al. 2006 'Robust Photometric Invariant Features from the Color Tensor'
def multi_gradient(imgs, sigma=0.5):
    N = len(imgs)
    # smooth images
    imgs = filters.gaussian_filter(imgs, sigma)
    # compute gradients
    gx, gy = gradient(imgs[0])
    # structure tensor (color tensor in RGB images)
    cxx = gx * gx
    cyy = gy * gy
    cxy = gx * gy
    # remaining channels
    for i in range(1, N):
        gx, gy = gradient(imgs[i])
        cxx += gx * gx
        cyy += gy * gy
        cxy += gx * gy
    cxy *= 2
    # smooth the structure/color tensor
    cxx = filters.gaussian_filter(cxx, sigma)
    cyy = filters.gaussian_filter(cyy, sigma)
    cxy = filters.gaussian_filter(cxy, sigma)
    cxx_cyy = cxx - cyy
    eps = 1e-9
    d = np.sqrt(cxx_cyy ** 2 + cxy ** 2 + eps)
    # largest eigenvalue - derivative energy in the most prominent direction
    lambda1 = cxx + cyy + d
    Gm = np.sqrt(lambda1 + eps)
    Gd = 0.5 * np.arctan2(cxy, cxx_cyy)
    return Gm, Gd


# gradient magnitude from an RGB image
def rgb_gradient(image, sigma=0.5):
    r = image[:, :, 0].astype("float32")
    g = image[:, :, 1].astype("float32")
    b = image[:, :, 2].astype("float32")

    imgs = []
    imgs.append(r)
    imgs.append(g)
    imgs.append(b)

    return multi_gradient(imgs, sigma)


def nonmaxsupress(gm, gd):
    """
    NMS based on the gradient magnitude (Gm) & gradient directions (Gd),
    returns gradient magnitude if local max, 0 otherwise
    """
    mask = np.pad(np.ones_like(gm[1:-1, 1:-1]), 1).astype(np.bool_)
    isobel = gm * np.sin(gd)
    jsobel = gm * np.cos(gd)
    local_maxima = _get_local_maxima(isobel, jsobel, gm, mask)
    local_maxima = np.where(local_maxima, gm, 0)
    return local_maxima


def hysteresisThreshold(nms, thLow, thHigh, binaryEdge=False):
    """
    Hysteresis thresholding based on NMS with gradient magnitude
    """
    low_mask = nms >= thLow
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask
    high_mask = nms >= thHigh
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]

    if binaryEdge:
        return 255 * (output_mask > 0)
    else:
        return np.where(output_mask, nms, 0)


def detect_rgb(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    binaryEdge=False,
    use_quantiles=True,
):
    Gm, Gd = rgb_gradient(image, sigma)
    low_threshold, high_threshold = np.percentile(
        Gm, [100.0 * low_threshold, 100.0 * high_threshold]
    )
    nms = nonmaxsupress(Gm, Gd)
    edge = hysteresisThreshold(nms, low_threshold, high_threshold, binaryEdge)
    return edge
