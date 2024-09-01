# ComfyUI-Color_Transfer

Implementation of color transfer using KMeans algorithm

## Installation

Clone repo to your custom_nodes folder
```git clone https://github.com/45uee/ComfyUI-Color_Transfer.git```

## Usage

1. Create a "Color Palette" node containing RGB values of your desired colors. Color must be defined in this format: [(Value, Value, Value), ...], for example [(30, 32, 30), (60, 61, 55), (105, 117, 101), (236, 223, 204)]
2. Create a "Palette Transfer" node, and connect your image and palette as input, that's all

## Example

![alt text](https://github.com/45uee/ComfyUI-Color_Transfer/blob/main/color_transfer_example.JPG)
