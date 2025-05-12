from .color_transfer import PaletteTransferNode, PalleteTransferClustering, PaletteTransferReinhard, PaletteSoftTransfer, PaletteRbfTransfer, PaletteOptimalTransportTransfer, ReferenceTransferReinhard, ColorPaletteNode


NODE_CLASS_MAPPINGS = {
    "PaletteTransfer": PaletteTransferNode,
    "PalleteTransferClustering": PalleteTransferClustering,
    "PaletteTransferReinhard": PaletteTransferReinhard,
    "PalletteSoftTransfer": PaletteSoftTransfer,
    "PaletteRbfTransfer": PaletteRbfTransfer,
    "PaletteOptimalTransportTransfer": PaletteOptimalTransportTransfer,
    "ColorTransferReinhard": ReferenceTransferReinhard,
    "ColorPalette": ColorPaletteNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteTransfer": "Palette Transfer",
}
