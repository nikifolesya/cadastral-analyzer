import math
import requests
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
from config import API_KEY, TILE_SIZE, DEFAULT_ZOOM, CROP_AREA

# Класс для загрузки спутниковых изображений через TomTom API
class SatelliteImageDownloader:
    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
    
    # Точное преобразование координат в тайловые координаты с дробными значениями    
    def lon_lat_to_tile_xy_precise(self, lon: float, lat: float, zoom: int) -> Tuple[float, float]:
        n = 2 ** zoom
        tile_x_precise = (lon + 180) / 360 * n
        lat_rad = math.radians(lat)
        tile_y_precise = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
        
        return tile_x_precise, tile_y_precise

    # Преобразование координат в целые номера тайлов
    def lon_lat_to_tile_xy(self, lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        tile_x_precise, tile_y_precise = self.lon_lat_to_tile_xy_precise(lon, lat, zoom)
        
        return int(tile_x_precise), int(tile_y_precise)

    # Загружает тайл через запрос к API
    def download_satellite_tile(self, zoom: int, x: int, y: int, format: str = 'jpg') -> Optional[Image.Image]:
        url = f"https://api.tomtom.com/map/1/tile/sat/main/{zoom}/{x}/{y}.{format}?key={self.api_key}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
            else:
                print(f"Ошибка {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Произошла ошибка при запросе: {e}")
            return None

    # Загружает изображение так, чтобы указанная точка была точно по центру
    def download_centered_satellite_image(self, lon: float, lat: float, zoom: int, 
                                        size_tiles: int = 3, format: str = 'jpg') -> Tuple[Optional[Image.Image], Tuple[float, float]]:
        
        # Получаем точные тайловые координаты
        center_x_precise, center_y_precise = self.lon_lat_to_tile_xy_precise(lon, lat, zoom)
        
        # Вычисляем какие тайлы нужно загрузить
        half_size = size_tiles // 2
        
        # Создаем большое изображение для сборки тайлов
        full_image = Image.new('RGB', (size_tiles * TILE_SIZE, size_tiles * TILE_SIZE))
        
        # Загружаем тайлы в сетку
        for i in range(size_tiles):
            for j in range(size_tiles):
                # Координаты тайла относительно центра
                tile_x = int(center_x_precise) + (i - half_size)
                tile_y = int(center_y_precise) + (j - half_size)
                
                # Загружаем тайл
                tile_image = self.download_satellite_tile(zoom, tile_x, tile_y, format)
                
                if tile_image:
                    # Вставляем тайл в нужную позицию
                    full_image.paste(tile_image, (i * TILE_SIZE, j * TILE_SIZE))
                else:
                    print(f"Не удалось загрузить тайл {tile_x}, {tile_y}")
        
        # Вычисляем точное смещение для центрирования
        center_pixel_x = (center_x_precise - int(center_x_precise) + half_size) * TILE_SIZE
        center_pixel_y = (center_y_precise - int(center_y_precise) + half_size) * TILE_SIZE
        
        # Размеры финального изображения
        final_width = TILE_SIZE * 2
        final_height = TILE_SIZE * 2
        
        # Вычисляем границы для кропа, чтобы точка была по центру
        left = int(center_pixel_x - final_width // 2)
        top = int(center_pixel_y - final_height // 2)
        right = left + final_width
        bottom = top + final_height
        
        # Обрезаем изображение так, чтобы точка была по центру
        centered_image = full_image.crop((left, top, right, bottom))
        
        return centered_image, (center_pixel_x, center_pixel_y)

    # Загружает изображение высокого разрешения с точной центровкой
    def download_high_resolution_image(self, lon: float, lat: float, 
                                     target_width: int = 1024, target_height: int = 1024, 
                                     zoom: int = DEFAULT_ZOOM, format: str = 'jpg') -> Optional[Image.Image]:
        
        # Вычисляем сколько тайлов нужно для покрытия желаемой области
        tiles_needed_x = math.ceil(target_width / TILE_SIZE) + 2
        tiles_needed_y = math.ceil(target_height / TILE_SIZE) + 2
        
        # Делаем размер нечетным для симметрии
        if tiles_needed_x % 2 == 0:
            tiles_needed_x += 1
        if tiles_needed_y % 2 == 0:
            tiles_needed_y += 1
        
        size_tiles = max(tiles_needed_x, tiles_needed_y)
        
        # Получаем изображение с центрированием
        centered_image, center_coords = self.download_centered_satellite_image(
            lon, lat, zoom, size_tiles, format
        )
        
        if centered_image is None:
            return None
        
        # Изменяем размер до нужных размеров
        final_image = centered_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Применяем кроп если необходимо
        final_image = final_image.crop(CROP_AREA)
        
        return final_image