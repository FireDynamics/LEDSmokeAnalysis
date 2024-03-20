def get_layer_from_height(low: float, high: float, value: float, nlayers: int) -> int:
    """Get layer for specific height above floor"""
    return int((high - value) / (high - low) * nlayers)


def get_height_from_layer(low: float, high: float, layer: float, nlayers: int) -> float:
    """Get height above floor for a specific layer"""
    return -1 * (layer / nlayers * (high - low) - high)


def get_widman_extco(l: int) -> float:
    """Get Widman relation for a specific wavelength"""
    return 4.8081 * l ** -1.0088
