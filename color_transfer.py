from itertools import combinations

import numpy as np
import torch
import ast
import cv2
import ot

from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import Delaunay
from skimage import color
from scipy.interpolate import Rbf

from comfy.comfy_types import IO, ComfyNodeABC

from .utils import (
    EuclideanDistance,
    ManhattanDistance,
    CosineSimilarity,
    HSVColorSimilarity,
    RGBWeightedDistance,
    RGBWeightedSimilarity,
    Blur,
)


class PaletteExtension:
    @staticmethod
    def dense_palette(
        base_palette: list[tuple[int, int, int]],
        points: int = 5,
        iterations: int = 2,
        extend_bw: bool = True,
    ) -> list[tuple[int, int, int]]:
        """
        Interpolate N points between each pair of colors in the base_palette.
        Do the same for the new colors generated, for a given number of iterations to
        create a dense mesh of interpolated colors.
        """
        # add black and white to the base palette
        if extend_bw:
            base_palette = set(base_palette)
            base_palette.add((0, 0, 0))
            base_palette.add((255, 255, 255))

        palette = np.array(list(base_palette), dtype=float)

        for _ in range(iterations):
            # build all combinations of two distinct indices
            idx_pairs = list(combinations(range(len(palette)), 2))

            # precompute interpolation fractions
            t = np.linspace(0, 1, points + 2)[1:-1]

            # for each pair, generate points intermediates
            new_colors = []
            for i, j in idx_pairs:
                c1, c2 = palette[i], palette[j]
                # broadcast interpolation in one go:
                inter = c1[None, :] + (c2 - c1)[None, :] * t[:, None]
                new_colors.append(inter)
            if new_colors:
                new_colors = np.vstack(new_colors)
                palette = np.vstack([palette, new_colors])

            # remove duplicates
            palette = np.unique(np.rint(palette).astype(int), axis=0).astype(float)

        # final unique, integer RGB list
        result = [tuple(rgb.astype(int)) for rgb in palette]
        return result

    @staticmethod
    def edge_based_palette(
        base_palette: list[tuple[int, int, int]], points: int = 5, iterations: int = 2
    ) -> list[tuple[int, int, int]]:
        """
        Use Delaunay triangulation to find edges between colors in the base_colors.
        Do the same for the new colors generated, for a given number of iterations to
        Create a dense mesh of interpolated colors.
        In contrast to dense_palette, this method uses the edges of the triangulation
        to generate new colors, which can lead to a more structured palette.
        """
        # add black and white to the base palette
        base_palette = set(base_palette)
        base_palette.add((0, 0, 0))
        base_palette.add((255, 255, 255))

        palette = np.array(list(base_palette), dtype=float)

        for _ in range(iterations):
            # Triangulate current palette to find adjacency edges
            if len(palette) >= palette.shape[1] + 1:
                tri = Delaunay(palette)
                edges = set()
                for simplex in tri.simplices:
                    # add all edges of the simplex
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            a, b = simplex[i], simplex[j]
                            edges.add(tuple(sorted((a, b))))
            else:
                # fallback to all pairs
                idxs = range(len(palette))
                edges = set(tuple(sorted((i, j))) for i in idxs for j in idxs if i < j)

            new_colors = []
            t = np.linspace(0, 1, points + 2)[1:-1][:, None]  # shape (points,1)

            for i, j in edges:
                c1, c2 = palette[i], palette[j]
                inters = c1 + (c2 - c1) * t  # (points,3)
                new_colors.append(inters)

            if new_colors:
                new_stack = np.vstack(new_colors)
                palette = np.vstack([palette, new_stack])
                # remove duplicates
                palette = np.unique(np.rint(palette).astype(int), axis=0).astype(float)

        return [tuple(c.astype(int)) for c in palette]


class ColorSpaceConvert:
    @staticmethod
    def convert_to_target_space(
        image: np.ndarray, target_colors: list[tuple[int, int, int]], color_space: str
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Convert image and target colors to specified color space."""
        if color_space == "RGB":
            return image, target_colors

        conversion_map = {
            "HSV": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "LAB": (cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB),
        }

        forward_conversion, _ = conversion_map[color_space]

        converted_image = cv2.cvtColor(image, forward_conversion)

        target_colors_array = np.array(target_colors, dtype=np.uint8).reshape(-1, 1, 3)
        converted_colors = cv2.cvtColor(target_colors_array, forward_conversion)
        converted_colors = [tuple(color[0]) for color in converted_colors]

        return converted_image, converted_colors

    @staticmethod
    def convert_to_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
        """Convert image back to RGB color space."""
        if color_space == "RGB":
            return image

        conversion_map = {"HSV": cv2.COLOR_HSV2RGB, "LAB": cv2.COLOR_LAB2RGB}

        return cv2.cvtColor(image, conversion_map[color_space])


class ColorClustering:
    def __init__(self, cluster_method: str):
        self.clustering_methods = {
            "Kmeans": KMeans,
            "Mini batch Kmeans": MiniBatchKMeans,
        }
        self.method = self.clustering_methods[cluster_method]

    def cluster_colors(self, image: np.ndarray, k: int) -> dict:
        """Perform color clustering on the image."""
        img_array = image.reshape((-1, 3))
        clustering_model = self.method(n_clusters=k, n_init="auto")
        clustering_model.fit(img_array)

        return {
            "image": image,
            "main_colors": clustering_model.cluster_centers_.astype(int),
            "model": clustering_model,
        }


class ColorMatcher:
    def __init__(self, distance_method: str):
        self.distance_methods = {
            "Euclidean": EuclideanDistance,
            "Manhattan": ManhattanDistance,
            "Cosine Similarity": CosineSimilarity,
            "HSV Distance": HSVColorSimilarity,
            "RGB Weighted Distance": RGBWeightedDistance,
            "RGB Weighted Similarity": RGBWeightedSimilarity,
        }
        self.distance_func = self.distance_methods[distance_method]

    def match_colors(
        self,
        detected_colors: np.ndarray,
        target_colors: list[tuple[int, int, int]],
        clustering_model: KMeans | MiniBatchKMeans,
        image_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Match detected colors with target colors using the specified distance method."""
        closest_colors = []

        for color in detected_colors:
            distances = self.distance_func(color, target_colors)
            closest_color = target_colors[np.argmin(distances)]
            closest_colors.append(closest_color)

        closest_colors = np.array(closest_colors)
        return closest_colors[clustering_model.labels_].reshape(image_shape)


class ImagePostProcessor:
    def __init__(self, gaussian_blur: int = 0):
        self.gaussian_blur = gaussian_blur

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Apply post-processing to the image."""
        processed = np.array(image).astype(np.float32)

        if self.gaussian_blur:
            processed = Blur(processed, self.gaussian_blur)

        return processed / 255.0


def process_image_with_palette(
    image: list[torch.Tensor],
    target_colors: list[tuple[int, int, int]],
    color_space: str,
    cluster_method: str,
    distance_method: str,
    gaussian_blur: int,
) -> torch.Tensor:
    """
    Shared function to process an image with the given parameters.
    """
    processedImages = []

    # Initialize components
    converter = ColorSpaceConvert()
    clustering_engine = ColorClustering(cluster_method)
    color_matcher = ColorMatcher(distance_method)
    image_processor = ImagePostProcessor(gaussian_blur)

    for img_tensor in image:
        # Prepare image
        img = 255.0 * img_tensor.cpu().numpy()

        # Convert color space
        converted_img, converted_colors = converter.convert_to_target_space(
            img, target_colors, color_space
        )

        # Perform clustering
        clustering_result = clustering_engine.cluster_colors(
            converted_img, len(target_colors)
        )

        # Match colors
        processed = color_matcher.match_colors(
            clustering_result["main_colors"],
            converted_colors,
            clustering_result["model"],
            converted_img.shape,
        )

        # Convert back to RGB
        processed = converter.convert_to_rgb(processed, color_space)

        # Post-process
        processed = image_processor.process_image(processed)
        processed_tensor = torch.from_numpy(processed)[None,]

        processedImages.append(processed_tensor)

    return torch.cat(processedImages, dim=0)


class ReferenceTransferReinhard(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "image_reference": (IO.IMAGE,),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Reference Transfer"

    def color_transfer(
        self, image: torch.Tensor, image_reference: torch.Tensor
    ) -> tuple[torch.Tensor]:

        processed_images = []

        # Handle reference image: flatten if it's a batch
        target = image_reference.cpu().numpy()
        if len(target.shape) == 4:  # If shape is (N, X, Y, 3)
            # Combine all N images into one big image for statistics calculation
            target = np.concatenate([target[i] for i in range(target.shape[0])], axis=0)

        for img_tensor in image:
            source = img_tensor.cpu().numpy()

            # Convert to Lab
            source_lab = color.rgb2lab(source)
            target_lab = color.rgb2lab(target)

            # Compute mean and std of each channel
            s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
            t_mean, t_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1))

            # Transfer color
            result_lab = (source_lab - s_mean) / s_std * t_std + t_mean
            result_rgb = np.clip(color.lab2rgb(result_lab), 0, 1)

            # Convert back to ComfyUI format
            # result_array = (result_rgb).astype(np.uint8)
            result_tensor = torch.from_numpy(result_rgb).unsqueeze(
                0
            )  # Add batch dimension

            processed_images.append(result_tensor)

        return (torch.cat(processed_images, dim=0),)


class PaletteOptimalTransportTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "palette_extension_method": (
                    ["Dense", "Edge", "None"],
                    {"default": "None"},
                ),
                "palette_extension_points": (
                    IO.INT,
                    {
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "default": 5,
                    },
                ),
                "blend_mode": (["Original", "Grayscale"], {"default": "Original"}),
                "blend_ratio": (
                    IO.FLOAT,
                    {
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "default": 0.5,
                    },
                ),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self,
        image: torch.Tensor,
        target_colors: list[tuple[int, int, int]],
        palette_extension_method: str,
        palette_extension_points: int,
        blend_mode: str,
        blend_ratio: float,
    ) -> tuple[torch.Tensor]:

        if palette_extension_method == "Dense":
            target_colors = PaletteExtension.dense_palette(
                target_colors, points=palette_extension_points
            )
        elif palette_extension_method == "Edge":
            target_colors = PaletteExtension.edge_based_palette(
                target_colors, points=palette_extension_points
            )

        palette = np.array(target_colors, dtype=np.float32) / 255.0
        n_palette = palette.shape[0]
        palette_weights = np.ones((n_palette,)) / n_palette

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy()
            h, w, _ = source.shape
            pixels = source.reshape(-1, 3)

            # KMeans clustering
            n_source_colors = 1000
            kmeans = MiniBatchKMeans(n_clusters=n_source_colors)
            kmeans.fit(pixels)
            source_centroids = kmeans.cluster_centers_
            pixel_labels = kmeans.labels_

            source_weights = np.bincount(pixel_labels) / len(pixel_labels)

            # Cost matrix (DO NOT normalize)
            cost_matrix = ot.dist(source_centroids, palette, metric="euclidean") ** 2

            # Compute OT transport plan
            transport_plan = ot.sinkhorn(
                source_weights,
                palette_weights,
                cost_matrix,
                reg=1e-2,
                numItermax=100000,
            )

            # Barycentric mapping (normalize per row)
            mapped_centroids = np.dot(transport_plan, palette) / np.sum(
                transport_plan, axis=1, keepdims=True
            )

            if blend_mode == "Original":
                # Blend between original and mapped colors
                recolored_pixels = (
                    1 - blend_ratio
                ) * pixels + blend_ratio * mapped_centroids[pixel_labels]
            elif blend_mode == "Grayscale":
                gray = color.rgb2gray(source)
                gray_rgb = np.stack([gray] * 3, axis=-1)
                gray_pixels = gray_rgb.reshape(-1, 3)
                # Blend between original and mapped colors
                recolored_pixels = (
                    1 - blend_ratio
                ) * gray_pixels + blend_ratio * mapped_centroids[pixel_labels]

            recolored_image = recolored_pixels.reshape(h, w, 3)

            result_tensor = torch.from_numpy(recolored_image).unsqueeze(0)
            processed_images.append(result_tensor)

        return (torch.cat(processed_images, dim=0),)


class PaletteRbfTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "rbf_function": (
                    ["thin_plate", "multiquadric", "inverse", "gaussian"],
                    {"default": "gaussian"},
                ),
                "epsilon": (
                    IO.FLOAT,
                    {"min": 0.01, "max": 100, "step": 0.1, "default": 1.0},
                ),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self,
        image: torch.Tensor,
        target_colors: list[tuple[int, int, int]],
        rbf_function: str,
        epsilon: float,
    ) -> tuple[torch.Tensor]:
        """
        Applies RBF interpolation to map image colors based on a given palette.

        Parameters:
        - image: Input image as a NumPy array in RGB format.
        - palette: List of RGB tuples representing the target palette.
        - rbf_function: Type of RBF ('thin_plate', 'multiquadric', 'inverse', 'gaussian', etc.).
        - epsilon: Adjustable constant for some RBF functions.

        Returns:
        - Recolored image as a NumPy array in RGB format.
        """
        # Extract R, G, B channels from the palette

        palette = np.array(target_colors, dtype=np.float32) / 255
        r, g, b = palette[:, 0], palette[:, 1], palette[:, 2]

        # Create RBF interpolators for each channel
        rbf_r = Rbf(r, g, b, r, function=rbf_function, epsilon=epsilon)
        rbf_g = Rbf(r, g, b, g, function=rbf_function, epsilon=epsilon)
        rbf_b = Rbf(r, g, b, b, function=rbf_function, epsilon=epsilon)

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy()
            h, w, _ = source.shape
            pixels = source.reshape(-1, 3)

            # Apply RBF interpolation to each pixel
            mapped_r = rbf_r(pixels[:, 0], pixels[:, 1], pixels[:, 2])
            mapped_g = rbf_g(pixels[:, 0], pixels[:, 1], pixels[:, 2])
            mapped_b = rbf_b(pixels[:, 0], pixels[:, 1], pixels[:, 2])

            # Stack and reshape the mapped channels
            mapped_pixels = np.stack((mapped_r, mapped_g, mapped_b), axis=-1)
            mapped_pixels = np.clip(mapped_pixels, 0, 1)
            recolored_image = mapped_pixels.reshape(h, w, 3)

            result_tensor = torch.from_numpy(recolored_image).unsqueeze(0)

            processed_images.append(result_tensor)

        return (torch.cat(processed_images, dim=0),)


class PaletteSoftTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "blend_mode": (["Original", "Grayscale"], {"default": "Original"}),
                "blend_ratio": (
                    IO.FLOAT,
                    {
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "default": 0.5,
                    },
                ),
                "softness": (
                    IO.FLOAT,
                    {
                        "min": 0,
                        "max": 20,
                        "step": 0.1,
                        "default": 1,
                    },
                ),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self,
        image: torch.Tensor,
        target_colors: list[tuple[int, int, int]],
        blend_mode: str,
        blend_ratio: float,
        softness: float,
    ) -> tuple[torch.Tensor]:
        """
        Shift image color mood towards N palette colors (soft harmonization)

        palette: list of N RGB colors [(R,G,B), ...]
        blend_ratio: how strongly to pull towards palette (0 = none, 1 = full)
        softness: how softly weights decay (higher = sharper attraction to nearest color)
        """
        if len(target_colors) < 2:
            raise ValueError("Palette must contain at least 2 colors")

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy()

            # Convert image + palette to Lab
            img_lab = color.rgb2lab(source)
            palette_lab = np.array(
                [color.rgb2lab(np.array([[c]]) / 255.0)[0, 0] for c in target_colors]
            )

            # Flatten image to pixels
            pixels = img_lab.reshape(-1, 3)

            # Compute distances to each palette color (Euclidean in Lab space)
            dists = np.array(
                [np.linalg.norm(pixels - p, axis=1) for p in palette_lab]
            )  # shape: (N_colors, N_pixels)

            # Convert distances to soft weights (inverse distance weighting)
            weights = np.exp(-softness * dists)
            weights /= weights.sum(axis=0)  # normalize to sum=1

            # Compute weighted average color for each pixel
            projected = np.tensordot(
                weights.T, palette_lab, axes=(1, 0)
            )  # shape: (N_pixels, 3)

            if blend_mode == "Original":
                # Blend between original and projected colors
                blended = (1 - blend_ratio) * pixels + blend_ratio * projected
            elif blend_mode == "Grayscale":
                gray = color.rgb2gray(source)  # shape: (H, W)
                gray_rgb = np.stack([gray] * 3, axis=-1)  # shape: (H, W, 3)
                # Convert grayscale RGB to Lab (so all blending happens in Lab)
                gray_lab = color.rgb2lab(gray_rgb)
                gray_pixels = gray_lab.reshape(-1, 3)
                # Blend between original and projected colors
                blended = (1 - blend_ratio) * gray_pixels + blend_ratio * projected

            # Reshape back and convert to RGB
            blended_lab = blended.reshape(img_lab.shape)
            blended_rgb = np.clip(color.lab2rgb(blended_lab), 0, 1)

            result_tensor = torch.from_numpy(blended_rgb).unsqueeze(
                0
            )  # Add batch dimension

            processed_images.append(result_tensor)

        return (torch.cat(processed_images, dim=0),)


class PaletteTransferReinhard(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self, image: torch.Tensor, target_colors: list[tuple[int, int, int]]
    ) -> tuple[torch.Tensor]:
        if len(target_colors) == 0:
            return (image,)

        target_colors = PaletteExtension.dense_palette(target_colors, points=3)

        def create_palette_image(
            palette: list[tuple[int, int, int]], size: tuple[int, int] = (1000, 1000)
        ) -> np.ndarray:
            """Creates a synthetic image from a list of RGB palette colors"""
            N = len(palette)
            width, height = size
            block_height = height // N

            img_array = np.zeros((height, width, 3), dtype=np.uint8)

            for i, color in enumerate(palette):
                img_array[i * block_height : (i + 1) * block_height, :] = color

            return img_array

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy()

            target = create_palette_image(target_colors) / 255.0

            # Convert to Lab
            source_lab = color.rgb2lab(source)
            target_lab = color.rgb2lab(target)

            # Compute mean and std of each channel
            s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
            t_mean, t_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1))

            # Transfer color
            result_lab = (source_lab - s_mean) / s_std * t_std + t_mean
            result_rgb = np.clip(color.lab2rgb(result_lab), 0, 1)

            result_tensor = torch.from_numpy(result_rgb).unsqueeze(
                0
            )  # Add batch dimension

            processed_images.append(result_tensor)

        return (torch.cat(processed_images, dim=0),)


class PalleteTransferClustering(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "palette_extension_method": (
                    ["Dense", "Edge", "None"],
                    {"default": "None"},
                ),
                "palette_extension_points": (
                    IO.INT,
                    {
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "default": 5,
                    },
                ),
                "gaussian_blur": (
                    IO.INT,
                    {"default": 3, "min": 0, "max": 27, "step": 1},
                ),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self,
        image: torch.Tensor,
        target_colors: list[tuple[int, int, int]],
        palette_extension_method: str,
        palette_extension_points: int,
        gaussian_blur: int,
    ) -> tuple[torch.Tensor]:
        if len(target_colors) == 0:
            return (image,)

        if palette_extension_method == "Dense":
            target_colors = PaletteExtension.dense_palette(
                target_colors, points=palette_extension_points
            )
        elif palette_extension_method == "Edge":
            target_colors = PaletteExtension.edge_based_palette(
                target_colors, points=palette_extension_points
            )

        output = process_image_with_palette(
            image=image,
            target_colors=target_colors,
            color_space="RGB",
            cluster_method="Mini batch Kmeans",
            distance_method="Euclidean",
            gaussian_blur=gaussian_blur,
        )

        return (output,)


class PaletteTransferNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "color_space": (["RGB", "HSV", "LAB"], {"default": "RGB"}),
                "cluster_method": (
                    ["Kmeans", "Mini batch Kmeans"],
                    {"default": "Kmeans"},
                ),
                "distance_method": (
                    [
                        "Euclidean",
                        "Manhattan",
                        "Cosine Similarity",
                        "HSV Distance",
                        "RGB Weighted Distance",
                        "RGB Weighted Similarity",
                    ],
                    {"default": "Euclidean"},
                ),
                "gaussian_blur": (
                    IO.INT,
                    {"default": 3, "min": 0, "max": 27, "step": 1},
                ),
            }
        }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self,
        image: torch.Tensor,
        target_colors: list[tuple[int, int, int]],
        color_space: str,
        cluster_method: str,
        distance_method: str,
        gaussian_blur: int,
    ) -> tuple[torch.Tensor]:
        if len(target_colors) == 0:
            return (image,)

        output = process_image_with_palette(
            image=image,
            target_colors=target_colors,
            color_space=color_space,
            cluster_method=cluster_method,
            distance_method=distance_method,
            gaussian_blur=gaussian_blur,
        )

        return (output,)


class ColorPaletteNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "color_palette": (
                    IO.STRING,
                    {
                        "default": "[(30, 32, 30), (60, 61, 55), (105, 117, 101), (236, 223, 204)]",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("COLOR_LIST",)
    RETURN_NAMES = ("Color palette",)
    FUNCTION = "color_list"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_list(self, color_palette: str) -> tuple[list[tuple[int, int, int]]]:
        return (ast.literal_eval(color_palette),)
