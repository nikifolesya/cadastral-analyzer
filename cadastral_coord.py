from rosreestr2coord.parser import Area
from shapely.geometry import Polygon

# Создание объекта Area с кадастровым номером участка
area = Area("38:06:144003:4723")

# Преобразование данных в формат GeoJSON
# area.to_geojson()

geometry = area.get_geometry()

coordinates = geometry['geometry']['coordinates'][0]

# Создаем объект Polygon
polygon = Polygon(coordinates)

# Находим центр участка
center = polygon.centroid
print(f"Координаты центра участка: {center.x}, {center.y}")

# # x_coords, y_coords = zip(*coordinates)
# # center_x, center_y = center.x, center.y

# # plt.figure(figsize=(8, 8))
# # plt.plot(x_coords, y_coords, 'b-', label='Межевание')
# # plt.fill(x_coords, y_coords, color='lightblue', alpha=0.3)
# # plt.scatter(center_x, center_y, color='red', label='Центр участка', zorder=5)
# # plt.text(center_x, center_y, " Центр", color="red", fontsize=10)

# # plt.title("Полигон участка и его центр")
# # plt.xlabel("Долгота")
# # plt.ylabel("Широта")
# # plt.legend()
# # plt.grid(True)
# # plt.axis('equal')
# # plt.show()
from dotenv import load_dotenv
from PIL import Image 
import os
import geemap
import ee
import requests
from io import BytesIO

load_dotenv()
KEY = os.getenv('KEY')
SERVICE_ACCOUNT = os.getenv('SERVICE_ACCOUNT')
PROJECT = os.getenv('PROJECT')

ee_creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY)
ee.Initialize(ee_creds)

# Создание карты с подложкой Google Satellite
Map = geemap.Map(center=[55.751244, 37.618423], zoom=17, basemap='SATELLITE')  # Москва

# Отобразить карту
Map
