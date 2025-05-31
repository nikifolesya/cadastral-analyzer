import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from typing import Optional
from config import DEVICE, MODEL_PATH


class ModelPredictor:
    """
    Класс для предсказания масок сегментации с помощью обученной модели
    """
    
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.transform = None
        self._init_model()
        self._init_transform()
    
    def _init_model(self):
        """Инициализация модели"""
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(self.device)
        
        try:
            if self.device == "cuda":
                self.model.load_state_dict(torch.load(self.model_path))
            else:
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            
            self.model.eval()
            print(f"Модель загружена успешно с устройства: {self.device}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            self.model = None
    
    def _init_transform(self):
        """Инициализация трансформаций для предобработки изображений"""
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def predict_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Предсказывает маску сегментации для изображения
        
        Args:
            image: входное изображение в формате numpy array (BGR)
            
        Returns:
            Бинарная маска или None при ошибке
        """
        
        if self.model is None:
            print("Модель не загружена")
            return None
        
        try:
            # Конвертируем из BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Применяем трансформации
            augmented = self.transform(image=image_rgb)
            input_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            # Делаем предсказание
            with torch.no_grad():
                output = self.model(input_tensor)
                output = torch.sigmoid(output)
                output = (output > 0.5).float()
            
            # Получаем маску
            pred_mask = output[0][0].cpu().numpy()
            
            # Изменяем размер маски до размера исходного изображения
            h, w = image.shape[:2]
            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            resized_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            return resized_mask
            
        except Exception as e:
            print(f"Ошибка при предсказании маски: {e}")
            return None
    
    def get_contours_from_mask(self, mask: np.ndarray, epsilon_factor: float = 0.022) -> list:
        """
        Находит контуры на основе маски
        
        Args:
            mask: бинарная маска
            epsilon_factor: параметр точности аппроксимации контура
            
        Returns:
            Список контуров
        """
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Аппроксимируем контуры
        approximated_contours = []
        for cnt in contours:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approximated_contours.append(approx)
        
        return approximated_contours
    
    def draw_contours_on_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                              color: tuple = (0, 255, 0), thickness: int = 5) -> np.ndarray:
        """
        Рисует контуры на изображении
        
        Args:
            image: исходное изображение
            mask: маска (если None, то предсказывается автоматически)
            color: цвет контуров
            thickness: толщина линий
            
        Returns:
            Изображение с нарисованными контурами
        """
        
        if mask is None:
            mask = self.predict_mask(image)
            if mask is None:
                return image
        
        # Получаем контуры
        contours = self.get_contours_from_mask(mask)
        
        # Копируем изображение
        result_image = image.copy()
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
        
        # Рисуем контуры
        for contour in contours:
            cv2.drawContours(result_image, [contour], -1, color, thickness)
        
        return result_image
    
    def create_colored_mask_overlay(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                                   mask_color: tuple = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
        """
        Создает наложение цветной маски на изображение
        
        Args:
            image: исходное изображение
            mask: маска (если None, то предсказывается автоматически)
            mask_color: цвет маски в формате BGR
            alpha: прозрачность маски (0-1)
            
        Returns:
            Изображение с наложенной маской
        """
        
        if mask is None:
            mask = self.predict_mask(image)
            if mask is None:
                return image
        
        # Создаем цветную маску
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = mask_color
        
        # Применяем маску с прозрачностью
        result_image = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return result_image