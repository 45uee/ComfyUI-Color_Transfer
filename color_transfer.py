import numpy as np
from sklearn.cluster import KMeans
import torch
import ast


def ColorClustering(image, k):
    img_array = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_array)
    main_colors = kmeans.cluster_centers_
    
    return image, main_colors.astype(int), kmeans


def SwitchColors(image, current_colors, target_colors, kmeans):
    closest_colors = []
    for color in current_colors:
        distances = np.linalg.norm(target_colors - color, axis=1)
        closest_color = target_colors[np.argmin(distances)]
        closest_colors.append(closest_color)
    closest_colors = np.array(closest_colors)

    image = closest_colors[kmeans.labels_].reshape(image.shape)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    
    return image


class PaletteTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": ("IMAGE",),
                "colors": ("COLORS",)
                }
            }
        return data_in

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"
    CATEGORY = "Palette Transfer"


    def color_transfer(self, image, colors):

        if len(colors) == 0:
            return (image,)
        else:
            processedImages = []

            for image in image:
                img = 255. * image.cpu().numpy()

                img, current_colors, kmeans = ColorClustering(img, len(colors))
                processed = SwitchColors(img, current_colors, colors, kmeans)
                processedImages.append(processed)
            output = torch.cat(processedImages, dim=0)

            return (output, )
    

class ColorPaletteNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colors": ("STRING", {'default': '', 'multiline': True})
            },
        }

    RETURN_TYPES = ("COLORS", )
    RETURN_NAMES = ("Color palette", )
    FUNCTION = "color_list"

    def color_list(self, colors):
        return (ast.literal_eval(colors), )