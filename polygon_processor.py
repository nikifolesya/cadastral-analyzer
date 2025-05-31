import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
from config import TILE_SIZE

# Класс для обработки и отрисовки полигонов на спутниковых изображениях
class PolygonProcessor:
    def __init__(self):
        pass
    
    # Преобразует географические координаты полигона в пиксельные координаты изображения
    def lon_lat_to_pixel_coords(self, polygon_coords: List[Tuple[float, float]], 
                               center_lon: float, center_lat: float, zoom: int, 
                               image_width: int, image_height: int, 
                               scale_factor: float = 1.0) -> List[Tuple[float, float]]:
        
        from image_downloader import SatelliteImageDownloader
        
        downloader = SatelliteImageDownloader()
        
        # Координаты центра в тайловой системе
        center_tile_x, center_tile_y = downloader.lon_lat_to_tile_xy_precise(center_lon, center_lat, zoom)
        
        # Центр изображения в пикселях
        center_pixel_x = image_width // 2
        center_pixel_y = image_height // 2
        
        pixel_coords = []
        
        for lon, lat in polygon_coords:
            # Преобразуем координату точки в тайловую систему
            point_tile_x, point_tile_y = downloader.lon_lat_to_tile_xy_precise(lon, lat, zoom)
            
            # Вычисляем смещение относительно центра в тайловых координатах
            delta_tile_x = point_tile_x - center_tile_x
            delta_tile_y = point_tile_y - center_tile_y
            
            # Преобразуем в пиксели с учетом масштабного коэффициента
            delta_pixel_x = delta_tile_x * TILE_SIZE * scale_factor
            delta_pixel_y = delta_tile_y * TILE_SIZE * scale_factor
            
            # Получаем финальные координаты в пикселях изображения
            pixel_x = center_pixel_x + delta_pixel_x
            pixel_y = center_pixel_y + delta_pixel_y
            
            pixel_coords.append((pixel_x, pixel_y))
        
        return pixel_coords

    # Рисует полигон на спутниковом изображении
    def draw_polygon_on_image(self, image: Image.Image, polygon_coords: List[Tuple[float, float]], 
                             center_lon: float, center_lat: float, zoom: int = 19, 
                             outline_color: str = 'red', outline_width: int = 8, 
                             fill_color: Optional[str] = None, scale_factor: float = 2) -> Image.Image:
        
        image_width, image_height = image.size
        
        # Преобразуем координаты полигона в пиксели
        pixel_coords = self.lon_lat_to_pixel_coords(
            polygon_coords, center_lon, center_lat, zoom, image_width, image_height, scale_factor
        )
        
        # Создаем объект для рисования
        draw = ImageDraw.Draw(image)
        
        # Рисуем полигон с жирным контуром
        if fill_color:
            draw.polygon(pixel_coords, outline=outline_color, fill=fill_color, width=outline_width)
        else:
            # Для более жирного контура рисуем несколько раз с небольшими смещениями
            for offset in range(-outline_width//2, outline_width//2 + 1):
                offset_coords = [(x + offset, y) for x, y in pixel_coords]
                draw.polygon(offset_coords, outline=outline_color)
                offset_coords = [(x, y + offset) for x, y in pixel_coords]
                draw.polygon(offset_coords, outline=outline_color)
        
        # Рисуем точки для отладки
        for x, y in pixel_coords:
            draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
        
        return image

    # Рисует полигон на изображении используя OpenCV
    def draw_polygon_opencv(self, image: np.ndarray, polygon_coords: List[Tuple[float, float]], 
                           center_lon: float, center_lat: float, zoom: int = 19, 
                           color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 3, 
                           scale_factor: float = 2) -> np.ndarray:
        
        result_image = image.copy()
        image_height, image_width = result_image.shape[:2]
        
        # Преобразуем координаты полигона в пиксели
        pixel_coords = self.lon_lat_to_pixel_coords(
            polygon_coords, center_lon, center_lat, zoom, image_width, image_height, scale_factor
        )
        
        # Преобразуем в формат для OpenCV
        polygon_points = np.array(pixel_coords, dtype=np.int32)
        
        # Рисуем полигон
        cv2.polylines(result_image, [polygon_points], True, color, thickness)
        
        # Рисуем точки полигона для отладки
        for x, y in pixel_coords:
            cv2.circle(result_image, (int(x), int(y)), 3, color, -1)
        
        return result_image
    
    # Создает бинарную маску полигона
    def create_polygon_mask(self, polygon_coords: List[Tuple[float, float]], 
                           center_lon: float, center_lat: float, zoom: int,
                           image_width: int, image_height: int, 
                           scale_factor: float = 2) -> np.ndarray:
        
        # Преобразуем координаты в пиксели
        pixel_coords = self.lon_lat_to_pixel_coords(
            polygon_coords, center_lon, center_lat, zoom, image_width, image_height, scale_factor
        )
        
        # Создаем маску
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        polygon_points = np.array(pixel_coords, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_points], 1)
        
        return mask