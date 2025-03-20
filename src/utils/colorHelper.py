import random
from src.config.graphConfig import GraphConfig
import numpy as np
class ColorHelper:
    @staticmethod
    def adjust_color(hex_color: str, factor: float) -> str:
        """
        Additional method for randomizing the color, either make it darker or lighter
        """
        
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))

        return f"#{r:02x}{g:02x}{b:02x}"
    
    def hex_to_rgba_tuple(hex_color: str, width: float = 1.0):
        """
        conver a format of #ffffef into rgbaw format
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = 255
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
        else:
            raise ValueError("Invalid hex color format")
        return (r, g, b, a, width)
    
    @staticmethod
    def create_lines_array(colors: list[str], default_width: float = 1.0):
        """
        Create a numpy array that consist of rgbaw from a list of colors. (mainly used for translating lines colors)
        """
        dtype = np.dtype([
            ('red', np.ubyte),
            ('green', np.ubyte),
            ('blue', np.ubyte),
            ('alpha', np.ubyte),
            ('width', float)
        ])
        lines_list = [ColorHelper.hex_to_rgba_tuple(color, default_width) for color in colors]
        lines = np.array(lines_list, dtype=dtype)
        return lines

    @staticmethod
    def generate_colors(num: int, mode: str = 'light') -> list[str]:
        """
        Generate list of colors randomly with length of num
        """

        if mode.lower() == 'light':
            base_colors = [
                GraphConfig.Colors.Light.red,
                GraphConfig.Colors.Light.green,
                GraphConfig.Colors.Light.blue,
                GraphConfig.Colors.Light.yellow,
                GraphConfig.Colors.Light.cyan,
                GraphConfig.Colors.Light.magenta
            ]
            
        else:
            base_colors = [
                GraphConfig.Colors.Dark.red,
                GraphConfig.Colors.Dark.green,
                GraphConfig.Colors.Dark.blue,
                GraphConfig.Colors.Dark.yellow,
                GraphConfig.Colors.Dark.cyan,
                GraphConfig.Colors.Dark.magenta
            ]
        base_count = len(base_colors)
        colors = []
        for i in range(num):
            base = base_colors[i % base_count]
            cycle = i // base_count 
            if mode.lower() == 'light':
                factor = 1 + (cycle * 0.1) 
            else:
                factor = 1 - (cycle * 0.1) 
                if factor < 0: 
                    factor = 0.1
           
            if random.random() < 0.2:
               
                factor *= random.uniform(0.9, 1.1)
            colors.append(ColorHelper.adjust_color(base, factor))
        return colors
