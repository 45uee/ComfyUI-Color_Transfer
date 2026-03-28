import gc
from itertools import combinations
import ast
import numpy as np
import torch
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
    MAX_PALETTE_SIZE = 512

    @staticmethod
    def dense_palette(
        base_palette: list[tuple[int, int, int]],
        points: int = 5,
        iterations: int = 2,
        extend_bw: bool = True,
    ) -> list[tuple[int, int, int]]:
        """
        Interpolate N points between each pair of colors in the base_palette.
        """
        if extend_bw:
            base_palette = set(base_palette)
            base_palette.add((0, 0, 0))
            base_palette.add((255, 255, 255))

        palette = np.array(list(base_palette), dtype=np.float32)

        for _ in range(iterations):
            if len(palette) > PaletteExtension.MAX_PALETTE_SIZE:
                break

            idx_pairs = list(combinations(range(len(palette)), 2))
            t = np.linspace(0, 1, points + 2)[1:-1].astype(np.float32)

            new_colors = []
            for i, j in idx_pairs:
                c1, c2 = palette[i], palette[j]
                inter = c1[None, :] + (c2 - c1)[None, :] * t[:, None]
                new_colors.append(inter)
            if new_colors:
                new_colors = np.vstack(new_colors)
                palette = np.vstack([palette, new_colors])

            palette = np.unique(np.rint(palette).astype(int), axis=0).astype(np.float32)

        result = [tuple(rgb.astype(int)) for rgb in palette]
        return result

    @staticmethod
    def edge_based_palette(
        base_palette: list[tuple[int, int, int]], points: int = 5, iterations: int = 2
    ) -> list[tuple[int, int, int]]:
        """
        Use Delaunay triangulation to find edges between colors.
        """
        base_palette = set(base_palette)
        base_palette.add((0, 0, 0))
        base_palette.add((255, 255, 255))

        palette = np.array(list(base_palette), dtype=np.float32)

        for _ in range(iterations):
            if len(palette) > PaletteExtension.MAX_PALETTE_SIZE:
                break

            if len(palette) >= palette.shape[1] + 1:
                try:
                    tri = Delaunay(palette)
                    edges = set()
                    for simplex in tri.simplices:
                        for i in range(len(simplex)):
                            for j in range(i + 1, len(simplex)):
                                a, b = simplex[i], simplex[j]
                                edges.add(tuple(sorted((a, b))))
                except Exception:
                    # Fallback if Delaunay fails (e.g. coplanar points)
                    idxs = range(len(palette))
                    edges = set(tuple(sorted((i, j))) for i in idxs for j in idxs if i < j)
            else:
                idxs = range(len(palette))
                edges = set(tuple(sorted((i, j))) for i in idxs for j in idxs if i < j)

            new_colors = []
            t = np.linspace(0, 1, points + 2)[1:-1][:, None].astype(np.float32)

            for i, j in edges:
                c1, c2 = palette[i], palette[j]
                inters = c1 + (c2 - c1) * t
                new_colors.append(inters)

            if new_colors:
                new_stack = np.vstack(new_colors)
                palette = np.vstack([palette, new_stack])
                palette = np.unique(np.rint(palette).astype(int), axis=0).astype(np.float32)

        return [tuple(c.astype(int)) for c in palette]


class ColorSpaceConvert:
    @staticmethod
    def convert_to_target_space(
        image: np.ndarray, target_colors: list[tuple[int, int, int]], color_space: str
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        if color_space == "RGB":
            return image, target_colors

        conversion_map = {
            "HSV": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "LAB": (cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB),
        }

        forward_conversion, _ = conversion_map[color_space]
        
        # Ensure image is float32 compliant for OpenCV conversions if needed, 
        # though cv2 usually handles standard types. keeping input type is safer.
        converted_image = cv2.cvtColor(image, forward_conversion)

        target_colors_array = np.array(target_colors, dtype=np.uint8).reshape(-1, 1, 3)
        converted_colors = cv2.cvtColor(target_colors_array, forward_conversion)
        converted_colors = [tuple(color[0]) for color in converted_colors]

        return converted_image, converted_colors

    @staticmethod
    def convert_to_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
        if color_space == "RGB":
            return image

        conversion_map = {"HSV": cv2.COLOR_HSV2RGB, "LAB": cv2.COLOR_LAB2RGB}
        # Ensure uint8 for consistent cvtColor behavior across color spaces
        return cv2.cvtColor(image.clip(0, 255).astype(np.uint8), conversion_map[color_space])


class ColorClustering:
    def __init__(self, cluster_method: str):
        self.clustering_methods = {
            "Kmeans": KMeans,
            "Mini batch Kmeans": MiniBatchKMeans,
        }
        self.method_name = cluster_method
        self.method = self.clustering_methods[cluster_method]

    def cluster_colors(self, image: np.ndarray, k: int) -> dict:
        """Perform color clustering on the image."""
        # Ensure we are working with float32 to save memory
        img_array = image.reshape((-1, 3)).astype(np.float32)
        
        # Add explicit parameters to control memory usage
        if self.method_name == "Mini batch Kmeans":
            clustering_model = self.method(
                n_clusters=k, 
                n_init="auto", 
                batch_size=4096, # Process in chunks
                reassignment_ratio=0 # Reduce bookkeeping
            )
        else:
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
        """Match detected colors with target colors."""
        closest_colors = []

        for color in detected_colors:
            distances = self.distance_func(color, target_colors)
            closest_color = target_colors[np.argmin(distances)]
            closest_colors.append(closest_color)

        closest_colors = np.array(closest_colors, dtype=np.float32)
        # Reconstruct image from labels
        return closest_colors[clustering_model.labels_].reshape(image_shape)


class ImagePostProcessor:
    def __init__(self, gaussian_blur: int = 0):
        self.gaussian_blur = gaussian_blur

    def process_image(self, image: np.ndarray) -> np.ndarray:
        processed = np.array(image, dtype=np.float32)

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

    converter = ColorSpaceConvert()
    clustering_engine = ColorClustering(cluster_method)
    color_matcher = ColorMatcher(distance_method)
    image_processor = ImagePostProcessor(gaussian_blur)

    for img_tensor in image:
        # Convert to uint8 0-255 so cv2.cvtColor handles all color spaces correctly
        img = (255.0 * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)

        converted_img, converted_colors = converter.convert_to_target_space(
            img, target_colors, color_space
        )

        clustering_result = clustering_engine.cluster_colors(
            converted_img, len(target_colors)
        )

        processed = color_matcher.match_colors(
            clustering_result["main_colors"],
            converted_colors,
            clustering_result["model"],
            converted_img.shape,
        )

        processed = converter.convert_to_rgb(processed, color_space)
        processed = image_processor.process_image(processed)
        
        # Ensure contiguous memory before converting to tensor
        processed = np.ascontiguousarray(processed)
        processed_tensor = torch.from_numpy(processed)[None,]
        processedImages.append(processed_tensor)

        # CLEANUP: Explicitly delete large temporary arrays and force GC
        del img
        del converted_img
        del clustering_result
        del processed
        
        # Only collect garbage if batch size is large to avoid perf hit on single images
        if len(image) > 1:
            gc.collect()

    return torch.cat(processedImages, dim=0)


class ReferenceTransferReinhard(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "image_reference": (IO.IMAGE,),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Reference Transfer"

    def color_transfer(
        self, image: torch.Tensor, image_reference: torch.Tensor
    ) -> tuple[torch.Tensor]:

        processed_images = []

        target = image_reference.cpu().numpy().astype(np.float32)
        if len(target.shape) == 4:
            target = np.concatenate([target[i] for i in range(target.shape[0])], axis=0)

        for img_tensor in image:
            source = img_tensor.cpu().numpy().astype(np.float32)

            source_lab = color.rgb2lab(source)
            target_lab = color.rgb2lab(target)

            s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
            t_mean, t_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1))

            # Avoid division by zero for uniform channels
            s_std = np.maximum(s_std, 1e-6)

            result_lab = (source_lab - s_mean) / s_std * t_std + t_mean
            result_rgb = np.clip(color.lab2rgb(result_lab), 0, 1)

            result_tensor = torch.from_numpy(result_rgb.astype(np.float32)).unsqueeze(0)
            processed_images.append(result_tensor)

            del source
            del source_lab
            del result_lab
            del result_rgb

        return (torch.cat(processed_images, dim=0),)


class PaletteOptimalTransportTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "palette_extension_method": (["Dense", "Edge", "None"], {"default": "None"}),
                "palette_extension_points": ("INT", {"min": 2, "max": 20, "step": 1, "default": 5}),
                "blend_mode": (["Original", "Grayscale"], {"default": "Original"}),
                "blend_ratio": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
            }
        }

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
            target_colors = PaletteExtension.dense_palette(target_colors, points=palette_extension_points)
        elif palette_extension_method == "Edge":
            target_colors = PaletteExtension.edge_based_palette(target_colors, points=palette_extension_points)

        palette = np.array(target_colors, dtype=np.float32) / 255.0
        n_palette = palette.shape[0]
        palette_weights = np.ones((n_palette,)) / n_palette

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy().astype(np.float32)
            h, w, _ = source.shape
            pixels = source.reshape(-1, 3)

            n_source_colors = 1000
            # MEMORY FIX: Reduce batch size and disable reassignment to save memory
            kmeans = MiniBatchKMeans(
                n_clusters=n_source_colors, 
                n_init="auto",
                batch_size=2048,
                reassignment_ratio=0
            )
            kmeans.fit(pixels)
            source_centroids = kmeans.cluster_centers_
            pixel_labels = kmeans.labels_

            source_weights = np.bincount(pixel_labels) / len(pixel_labels)

            cost_matrix = ot.dist(source_centroids, palette, metric="euclidean") ** 2

            transport_plan = ot.sinkhorn(
                source_weights,
                palette_weights,
                cost_matrix,
                reg=1e-2,
                numItermax=100000,
            )

            mapped_centroids = np.dot(transport_plan, palette) / np.sum(
                transport_plan, axis=1, keepdims=True
            )

            if blend_mode == "Original":
                recolored_pixels = (1 - blend_ratio) * pixels + blend_ratio * mapped_centroids[pixel_labels]
            elif blend_mode == "Grayscale":
                gray = color.rgb2gray(source)
                gray_rgb = np.stack([gray] * 3, axis=-1)
                gray_pixels = gray_rgb.reshape(-1, 3)
                recolored_pixels = (1 - blend_ratio) * gray_pixels + blend_ratio * mapped_centroids[pixel_labels]

            recolored_image = recolored_pixels.reshape(h, w, 3).astype(np.float32)

            result_tensor = torch.from_numpy(recolored_image).unsqueeze(0)
            processed_images.append(result_tensor)

            # CLEANUP
            del kmeans
            del source
            del pixels
            del cost_matrix
            del transport_plan
            
            if len(image) > 1:
                gc.collect()

        return (torch.cat(processed_images, dim=0),)


class PaletteRbfTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "rbf_function": (["thin_plate", "multiquadric", "inverse", "gaussian"], {"default": "gaussian"}),
                "epsilon": ("FLOAT", {"min": 0.01, "max": 100, "step": 0.1, "default": 1.0}),
            }
        }

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
        
        palette = np.array(target_colors, dtype=np.float32) / 255
        r, g, b = palette[:, 0], palette[:, 1], palette[:, 2]

        rbf_r = Rbf(r, g, b, r, function=rbf_function, epsilon=epsilon)
        rbf_g = Rbf(r, g, b, g, function=rbf_function, epsilon=epsilon)
        rbf_b = Rbf(r, g, b, b, function=rbf_function, epsilon=epsilon)

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy().astype(np.float32)
            h, w, _ = source.shape
            pixels = source.reshape(-1, 3)

            mapped_r = rbf_r(pixels[:, 0], pixels[:, 1], pixels[:, 2])
            mapped_g = rbf_g(pixels[:, 0], pixels[:, 1], pixels[:, 2])
            mapped_b = rbf_b(pixels[:, 0], pixels[:, 1], pixels[:, 2])

            mapped_pixels = np.stack((mapped_r, mapped_g, mapped_b), axis=-1)
            mapped_pixels = np.clip(mapped_pixels, 0, 1).astype(np.float32)
            recolored_image = mapped_pixels.reshape(h, w, 3)

            result_tensor = torch.from_numpy(recolored_image).unsqueeze(0)
            processed_images.append(result_tensor)
            
            del source
            del pixels
            del mapped_pixels

        return (torch.cat(processed_images, dim=0),)


class PaletteSoftTransfer(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "blend_mode": (["Original", "Grayscale"], {"default": "Original"}),
                "blend_ratio": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
                "softness": ("FLOAT", {"min": 0, "max": 20, "step": 0.1, "default": 1}),
            }
        }

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
        
        if len(target_colors) < 2:
            raise ValueError("Palette must contain at least 2 colors")

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy().astype(np.float32)

            img_lab = color.rgb2lab(source)
            palette_lab = np.array(
                [color.rgb2lab(np.array([[c]]) / 255.0)[0, 0] for c in target_colors]
            )

            pixels = img_lab.reshape(-1, 3).astype(np.float32)
            n_pixels = pixels.shape[0]
            n_colors = len(palette_lab)
            chunk_size = max(1, 500_000 // n_colors)

            projected = np.empty_like(pixels)
            for start in range(0, n_pixels, chunk_size):
                end = min(start + chunk_size, n_pixels)
                chunk = pixels[start:end]
                # (n_colors, chunk_len)
                dists = np.array([np.linalg.norm(chunk - p, axis=1) for p in palette_lab])
                weights = np.exp(-softness * dists)
                weights /= weights.sum(axis=0)
                projected[start:end] = np.dot(weights.T, palette_lab)

            if blend_mode == "Original":
                blended = (1 - blend_ratio) * pixels + blend_ratio * projected
            elif blend_mode == "Grayscale":
                gray = color.rgb2gray(source)
                gray_rgb = np.stack([gray] * 3, axis=-1)
                gray_lab = color.rgb2lab(gray_rgb)
                gray_pixels = gray_lab.reshape(-1, 3)
                blended = (1 - blend_ratio) * gray_pixels + blend_ratio * projected

            blended_lab = blended.reshape(img_lab.shape)
            blended_rgb = np.clip(color.lab2rgb(blended_lab), 0, 1).astype(np.float32)

            result_tensor = torch.from_numpy(blended_rgb).unsqueeze(0)
            processed_images.append(result_tensor)
            
            del source
            del img_lab
            del pixels

        return (torch.cat(processed_images, dim=0),)


class PaletteTransferReinhard(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Color Transfer/Palette Transfer"

    def color_transfer(
        self, image: torch.Tensor, target_colors: list[tuple[int, int, int]]
    ) -> tuple[torch.Tensor]:
        if len(target_colors) == 0:
            return (image,)

        target_colors = PaletteExtension.dense_palette(target_colors, points=3)

        # Compute target LAB stats directly from palette instead of creating a large image
        palette_rgb = np.array(target_colors, dtype=np.float32).reshape(-1, 1, 3) / 255.0
        palette_lab = color.rgb2lab(palette_rgb).reshape(-1, 3)
        t_mean = palette_lab.mean(axis=0)
        t_std = np.maximum(palette_lab.std(axis=0), 1e-6)

        processed_images = []

        for img_tensor in image:
            source = img_tensor.cpu().numpy().astype(np.float32)

            source_lab = color.rgb2lab(source)

            s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))

            # Avoid division by zero for uniform channels
            s_std = np.maximum(s_std, 1e-6)

            result_lab = (source_lab - s_mean) / s_std * t_std + t_mean
            result_rgb = np.clip(color.lab2rgb(result_lab), 0, 1).astype(np.float32)

            result_tensor = torch.from_numpy(result_rgb).unsqueeze(0)
            processed_images.append(result_tensor)

            del source
            del source_lab

        return (torch.cat(processed_images, dim=0),)


class PalleteTransferClustering(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "palette_extension_method": (["Dense", "Edge", "None"], {"default": "None"}),
                "palette_extension_points": ("INT", {"min": 2, "max": 20, "step": 1, "default": 5}),
                "gaussian_blur": ("INT", {"default": 3, "min": 0, "max": 27, "step": 1}),
            }
        }

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
            target_colors = PaletteExtension.dense_palette(target_colors, points=palette_extension_points)
        elif palette_extension_method == "Edge":
            target_colors = PaletteExtension.edge_based_palette(target_colors, points=palette_extension_points)

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
        return {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "color_space": (["RGB", "HSV", "LAB"], {"default": "RGB"}),
                "cluster_method": (["Kmeans", "Mini batch Kmeans"], {"default": "Kmeans"}),
                "distance_method": ([
                        "Euclidean",
                        "Manhattan",
                        "Cosine Similarity",
                        "HSV Distance",
                        "RGB Weighted Distance",
                        "RGB Weighted Similarity",
                    ], {"default": "Euclidean"}),
                "gaussian_blur": ("INT", {"default": 3, "min": 0, "max": 27, "step": 1}),
            }
        }

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
        parsed = ast.literal_eval(color_palette)
        if not isinstance(parsed, list) or not all(
            isinstance(c, tuple) and len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)
            for c in parsed
        ):
            raise ValueError("Color palette must be a list of (R, G, B) tuples with values 0-255")
        return (parsed,)


class ExtractPaletteNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (IO.IMAGE,),
                "num_colors": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "cluster_method": (["Kmeans", "Mini batch Kmeans"], {"default": "Mini batch Kmeans"}),
            },
        }

    RETURN_TYPES = ("COLOR_LIST",)
    RETURN_NAMES = ("Color palette",)
    FUNCTION = "extract_palette"
    CATEGORY = "Color Transfer/Palette Transfer"

    def extract_palette(
        self, image: torch.Tensor, num_colors: int, cluster_method: str
    ) -> tuple[list[tuple[int, int, int]]]:
        
        img_tensor = image[0] if len(image.shape) == 4 else image
        
        # MEMORY FIX: Ensure float32 for clustering
        img = (255.0 * img_tensor.cpu().numpy()).astype(np.float32)
        
        clustering_engine = ColorClustering(cluster_method)
        clustering_result = clustering_engine.cluster_colors(img, num_colors)
        
        colors = clustering_result["main_colors"]
        color_list = [tuple(map(int, color)) for color in colors]
        
        # Cleanup
        del img
        del clustering_result
        
        return (color_list,)