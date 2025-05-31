import os
import re
import dotenv
import torch

# Загружаем переменные окружения
dotenv.load_dotenv()

# Константы
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv('TOMTOM_API_KEY')
TILE_SIZE = 256
MODEL_PATH = "model/unet_resnet34_plot_segmentation.pth"

# Регулярное выражение для проверки кадастрового номера
CADASTRAL_NUMBER_REGEX = r'^\d{1,2}:\d{1,2}:(\d|\d{6,7}):\d{1,10}$'

# Настройки изображения
DEFAULT_ZOOM = 19
DEFAULT_TARGET_WIDTH = 1024
DEFAULT_TARGET_HEIGHT = 1024
CROP_AREA = (50, 100, 974, 924)

# Проверяет корректность кадастрового номера
def validate_cadastral_number(cadastral_number: str) -> bool:
    if not cadastral_number:
        return False
    
    return bool(re.match(CADASTRAL_NUMBER_REGEX, cadastral_number.strip()))