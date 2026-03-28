import numpy as np
import cv2


def EuclideanDistance(detected_color, target_colors):
    return np.linalg.norm(detected_color - target_colors, axis=1)


def ManhattanDistance(detected_color, target_colors):
    return np.sum(np.abs(detected_color - target_colors), axis=1)


def CosineSimilarity(detected_color, target_colors):
    return -np.dot(target_colors, detected_color) / (np.linalg.norm(detected_color) * np.linalg.norm(target_colors, axis=1))


def RGBWeightedDistance(detected_color, target_colors):
    detected_color = np.array(detected_color, dtype=np.float64)
    target_colors = np.array(target_colors, dtype=np.float64)

    weights = np.array([0.299, 0.587, 0.114])

    diff = detected_color - target_colors
    return np.sqrt(np.sum(weights * diff ** 2, axis=1))


def RGBWeightedSimilarity(detected_color, target_colors):
    detected_color = np.array(detected_color, dtype=np.float64)
    target_colors = np.array(target_colors, dtype=np.float64)

    weights = np.sqrt(np.array([0.299, 0.587, 0.114]))

    weighted_detected = detected_color * weights
    weighted_targets = target_colors * weights

    dot_products = np.dot(weighted_targets, weighted_detected)
    norm1 = np.linalg.norm(weighted_detected)
    norm2 = np.linalg.norm(weighted_targets, axis=1)

    denom = norm1 * norm2
    denom = np.maximum(denom, 1e-10)

    return -dot_products / denom


def HSVColorSimilarity(detected_color, target_colors):
    detected_color = np.array(detected_color)
    target_colors = np.array(target_colors)
    
    h1, s1, _ = detected_color
    h2 = target_colors[:, 0]
    s2 = target_colors[:, 1]

    h1_rad = np.radians(h1)
    h2_rad = np.radians(h2)
    
    v1_x = s1 * np.cos(h1_rad)
    v1_y = s1 * np.sin(h1_rad)
    v1 = np.array([v1_x, v1_y])
    

    v2_x = s2 * np.cos(h2_rad)
    v2_y = s2 * np.sin(h2_rad)
    v2 = np.vstack([v2_x, v2_y])
    
    dot_products = np.dot(v1, v2)

    v1_norm = np.linalg.norm(v1)
    v2_norms = np.linalg.norm(v2, axis=0)

    denom = v1_norm * v2_norms
    denom = np.maximum(denom, 1e-10)

    similarities = dot_products / denom

    return -similarities
    

def Blur(image, kernel_size):
    return cv2.medianBlur(image.astype(np.uint8), kernel_size).astype(np.float32)