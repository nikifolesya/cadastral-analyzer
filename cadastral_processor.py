import numpy as np
import cv2
from PIL import Image
from rosreestr2coord.parser import Area
from shapely.geometry import Polygon
from typing import Dict, Tuple, Optional
import os

from image_downloader import SatelliteImageDownloader
from polygon_processor import PolygonProcessor
from model_predictor import ModelPredictor
from config import DEFAULT_ZOOM, DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT


class CadastralProcessor:
    """
    Главный класс для обработки кадастровых участков
    """
    
    def __init__(self):
        self.downloader = SatelliteImageDownloader()
        self.polygon_processor = PolygonProcessor()
        self.model_predictor = ModelPredictor()
        
    def get_cadastral_data(self, cadastral_number: str) -> Dict:
        """
        Получает данные кадастрового участка
        
        Args:
            cadastral_number: кадастровый номер
            
        Returns:
            Словарь с данными участка
        """
        try:
            area = Area(cadastral_number)
            geometry = area.get_geometry()
            coordinates = geometry['geometry']['coordinates'][0]
            
            # Создаем полигон и находим центр
            polygon = Polygon(coordinates)
            center = polygon.centroid
            center_lon, center_lat = center.x, center.y
            
            return {
                'coordinates': coordinates,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'polygon': polygon,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def download_satellite_image(self, center_lon: float, center_lat: float) -> Optional[Image.Image]:
        """
        Загружает спутниковое изображение для заданных координат
        
        Args:
            center_lon, center_lat: координаты центра
            
        Returns:
            PIL Image или None
        """
        try:
            image = self.downloader.download_high_resolution_image(
                center_lon, center_lat,
                target_width=DEFAULT_TARGET_WIDTH,
                target_height=DEFAULT_TARGET_HEIGHT,
                zoom=DEFAULT_ZOOM
            )
            return image
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return None
    
    def save_image(self, image: Image.Image, center_lat: float, center_lon: float) -> str:
        """
        Сохраняет изображение и возвращает путь к файлу
        
        Args:
            image: PIL изображение
            center_lat, center_lon: координаты центра
            
        Returns:
            Путь к сохраненному файлу
        """
        filename = f"{center_lat},{center_lon}.png"
        image.save(filename, quality=95)
        return filename
    
    def process_with_mask_and_polygon(self, cadastral_number: str, 
                                    show_mask: bool = True, 
                                    show_contour: bool = True, 
                                    show_polygon: bool = True) -> Dict:
        """
        Полная обработка кадастрового участка с отрисовкой маски, контура и полигона
        
        Args:
            cadastral_number: кадастровый номер участка
            show_mask: отображать ли маску сегментации
            show_contour: отображать ли контур, найденный нейросетью
            show_polygon: отображать ли кадастровый полигон
            
        Returns:
            Словарь с результатами обработки
        """
        
        # Получаем данные участка
        cadastral_data = self.get_cadastral_data(cadastral_number)
        if not cadastral_data['success']:
            return {'success': False, 'error': cadastral_data['error']}
        
        coordinates = cadastral_data['coordinates']
        center_lon = cadastral_data['center_lon']
        center_lat = cadastral_data['center_lat']
        
        # Загружаем спутниковое изображение
        satellite_image = self.download_satellite_image(center_lon, center_lat)
        if satellite_image is None:
            return {'success': False, 'error': 'Не удалось загрузить спутниковое изображение'}
        
        # Сохраняем исходное изображение
        image_filename = self.save_image(satellite_image, center_lat, center_lon)
        
        # Конвертируем в numpy array для работы с OpenCV
        image_cv = cv2.cvtColor(np.array(satellite_image), cv2.COLOR_RGB2BGR)
        result_image = image_cv.copy()
        
        # Получаем маску предсказания
        mask = None
        if show_mask or show_contour:
            mask = self.model_predictor.predict_mask(image_cv)
            if mask is None:
                return {'success': False, 'error': 'Не удалось получить предсказание модели'}
        
        # 1. Отрисовка маски сегментации (если включена)
        if show_mask and mask is not None:
            result_image = self.model_predictor.create_colored_mask_overlay(
                result_image, mask, mask_color=(0, 0, 255), alpha=0.3
            )
        
        # 2. Отрисовка контура, найденного нейросетью (если включена)
        if show_contour and mask is not None:
            result_image = self.model_predictor.draw_contours_on_image(
                result_image, mask, color=(0, 255, 0), thickness=5
            )
        
        # 3. Отрисовка кадастрового полигона (если включена)
        if show_polygon:
            result_image = self.polygon_processor.draw_polygon_opencv(
                result_image, coordinates, center_lon, center_lat, 
                zoom=DEFAULT_ZOOM, color=(255, 0, 0), thickness=3, scale_factor=2
            )
        
        # Конвертируем обратно в PIL для возврата
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Сохраняем результат
        result_filename = f"cadastral_{cadastral_number.replace(':', '_')}_result.png"
        result_image_pil.save(result_filename)
        
        return {
            'success': True,
            'original_image': satellite_image,
            'result_image': result_image_pil,
            'original_filename': image_filename,
            'result_filename': result_filename,
            'center_coords': (center_lon, center_lat),
            'coordinates': coordinates,
            'mask': mask
        }
    
    def analyze_mask_polygon_overlap(self, cadastral_number: str) -> Dict:
        """
        Анализирует пересечение между маской сегментации и кадастровым полигоном
        
        Args:
            cadastral_number: кадастровый номер
            
        Returns:
            Словарь с метриками пересечения
        """
        
        # Получаем данные участка
        cadastral_data = self.get_cadastral_data(cadastral_number)
        if not cadastral_data['success']:
            return {'success': False, 'error': cadastral_data['error']}
        
        coordinates = cadastral_data['coordinates']
        center_lon = cadastral_data['center_lon']
        center_lat = cadastral_data['center_lat']
        
        # Загружаем изображение
        satellite_image = self.download_satellite_image(center_lon, center_lat)
        if satellite_image is None:
            return {'success': False, 'error': 'Не удалось загрузить спутниковое изображение'}
        
        # Конвертируем в numpy array
        image_cv = cv2.cvtColor(np.array(satellite_image), cv2.COLOR_RGB2BGR)
        
        # Получаем маску предсказания
        predicted_mask = self.model_predictor.predict_mask(image_cv)
        if predicted_mask is None:
            return {'success': False, 'error': 'Не удалось получить предсказание модели'}
        
        # Создаем маску кадастрового полигона
        image_height, image_width = image_cv.shape[:2]
        polygon_mask = self.polygon_processor.create_polygon_mask(
            coordinates, center_lon, center_lat, DEFAULT_ZOOM, 
            image_width, image_height, scale_factor=2
        )
        
        # Вычисляем метрики пересечения
        intersection = np.logical_and(predicted_mask, polygon_mask)
        union = np.logical_or(predicted_mask, polygon_mask)
        
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        precision = np.sum(intersection) / np.sum(predicted_mask) if np.sum(predicted_mask) > 0 else 0
        recall = np.sum(intersection) / np.sum(polygon_mask) if np.sum(polygon_mask) > 0 else 0
        
        return {
            'success': True,
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'mask_area': int(np.sum(predicted_mask)),
            'polygon_area': int(np.sum(polygon_mask)),
            'intersection_area': int(np.sum(intersection))
        }
    
    def cleanup_files(self, filenames: list):
        """
        Удаляет временные файлы
        
        Args:
            filenames: список имен файлов для удаления
        """
        for filename in filenames:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(f"Не удалось удалить файл {filename}: {e}")