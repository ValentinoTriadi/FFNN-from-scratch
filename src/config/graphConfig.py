class GraphConfig:

    # Performance
    TEXT_SIZE = 12
    NODE_SIZE = 50
    LAYER_SPACING = 50
    MAX_EDGE_DISTANCE = 55
    LAYER_Y_RANGE = (0, 120)
    LINE_SIZE = 1

    # Style
    TEXT_COLOR = "#ffffff"
    BACKGROUND_COLOR = "#000000"
    WEIGHT_TABLE_ROWS = 10
    MARGIN_RIGHT = 200
    class Colors:
        class Light:
            red = "#ff4d4d"
            green = "#4dff88"
            blue = "#00ffaa"
            yellow = "#ffea00"
            cyan = "#00e0e0"
            magenta = "#ff00ff"
        class Dark:
            red = "#cc0000"
            green = "#009933"
            blue = "#007f7f"
            yellow = "#cc9900"
            cyan = "#007777"
            magenta = "#cc00cc"