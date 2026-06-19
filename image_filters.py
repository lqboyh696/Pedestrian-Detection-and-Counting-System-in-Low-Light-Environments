# image_filters.py - 智能后处理滤镜模块
# 用于在低光照增强后进行高光保护融合，防止过曝

import cv2
import numpy as np


def apply_highlight_protection(orig_frame, enhanced_frame):
    """
    智能高光保护融合 (基于亮度平方蒙版)
    原理：原始帧中较亮的区域（如灯光、天空）保留更多原始像素，
          增强帧主要作用于暗部区域，避免亮部过曝泛白
    """
    if orig_frame is None or enhanced_frame is None:
        return enhanced_frame

    # 确保增强后图像与原始图像尺寸一致
    if orig_frame.shape != enhanced_frame.shape:
        orig_frame = cv2.resize(orig_frame, (enhanced_frame.shape[1], enhanced_frame.shape[0]))

    # 计算原始图像的亮度蒙版：越亮的像素蒙版值越大(接近1)
    gray_orig = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    mask = (gray_orig.astype(np.float32) / 255.0) ** 2  # 平方增强亮暗差异
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 加权融合：亮区保留原图，暗区使用增强结果
    final_img = orig_frame.astype(np.float32) * mask + enhanced_frame.astype(np.float32) * (1 - mask)
    return np.clip(final_img, 0, 255).astype(np.uint8)