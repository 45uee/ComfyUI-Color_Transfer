from .color_transfer import PaletteTransferNode, ColorPaletteNode


NODE_CLASS_MAPPINGS = {
    "PaletteTransfer": PaletteTransferNode,
    "ColorPalette": ColorPaletteNode,   
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteTransfer": "Palette Transfer",
    "ColorPalette": "Color Palette",
}
