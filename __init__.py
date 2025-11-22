from .color_transfer import PaletteTransferNode, PalleteTransferClustering, PaletteTransferReinhard, PaletteSoftTransfer, PaletteRbfTransfer, PaletteOptimalTransportTransfer, ReferenceTransferReinhard, ColorPaletteNode, ExtractPaletteNode


NODE_CLASS_MAPPINGS = {
    "PaletteTransfer": PaletteTransferNode,
    "PalleteTransferClustering": PalleteTransferClustering,
    "PaletteTransferReinhard": PaletteTransferReinhard,
    "PalletteSoftTransfer": PaletteSoftTransfer,
    "PaletteRbfTransfer": PaletteRbfTransfer,
    "PaletteOptimalTransportTransfer": PaletteOptimalTransportTransfer,
    "ColorTransferReinhard": ReferenceTransferReinhard,
    "ColorPalette": ColorPaletteNode,
    "ExtractPalette": ExtractPaletteNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteTransfer": "Palette Transfer",
}
