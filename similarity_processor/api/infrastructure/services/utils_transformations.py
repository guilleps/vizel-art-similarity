import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def apply_contrast_enhancement(image):
    """Apply contrast enhancement to the input image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    logger.info('Applied transformation: contrast enhancement')
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def apply_texture_direction(image):
    """Extract texture direction using Sobel operators."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    logger.info('Applied transformation: texture direction')
    return magnitude.astype(np.uint8)

def apply_color_distribution_map(image):
    """Generate a color distribution heatmap."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap = np.mean(img_rgb, axis=2)  # Average across channels to estimate "density"
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    logger.info('Applied transformation: color distribution map')
    return plt.cm.plasma(heatmap.astype(np.uint8))

def apply_hsv_channels(image, delta_h=15, factor_s=1.20, factor_v=1.15):
    """Extract and normalize individual HSV channels."""
    if not (-30 <= int(delta_h) <= 30):
        raise ValueError("TT (tone) parameter out of range: delta_h must be in [-30, 30].")
    if not (0.50 <= float(factor_s) <= 1.50):
        raise ValueError("TS (saturation) parameter out of range: factor_s must be in [0.50, 1.50].")
    if not (0.60 <= float(factor_v) <= 1.40):
        raise ValueError("TB (brightness) parameter out of range: factor_v must be in [0.60, 1.40].")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # TT
    h_tt = ((h.astype(np.int16) + int(delta_h)) % 180).astype(np.uint8)
    hsv_tt = cv2.merge((h_tt, s, v))
    tt_img = cv2.cvtColor(hsv_tt, cv2.COLOR_HSV2BGR)

    # TS
    s_ts = np.clip(s.astype(np.float32) * float(factor_s), 0, 255).astype(np.uint8)
    hsv_ts = cv2.merge((h, s_ts, v))
    ts_img = cv2.cvtColor(hsv_ts, cv2.COLOR_HSV2BGR)

    # TB
    v_tb = np.clip(v.astype(np.float32) * float(factor_v), 0, 255).astype(np.uint8)
    hsv_tb = cv2.merge((h, s, v_tb))
    tb_img = cv2.cvtColor(hsv_tb, cv2.COLOR_HSV2BGR)

    logger.info('Applied transformation: HSV channels')
    return {
        "hue": tt_img,
        "saturation": ts_img,
        "value": tb_img
    }
