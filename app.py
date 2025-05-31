import streamlit as st
import pandas as pd

from config import validate_cadastral_number
from cadastral_processor import CadastralProcessor

def main():
    st.set_page_config(
        page_title="Анализ кадастровых участков",
        layout="wide"
    )
    
    st.title("Анализ кадастровых участков")
    st.markdown("Система анализа кадастровых участков с использованием спутниковых снимков и машинного обучения")
    
    # Инициализация процессора
    if 'processor' not in st.session_state:
        with st.spinner('Инициализация системы...'):
            st.session_state.processor = CadastralProcessor()
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        
        # Настройки отображения
        st.subheader("Отображение")
        show_mask = st.checkbox("Показать маску сегментации", value=True)
        show_contour = st.checkbox("Показать контур (нейросеть)", value=True)
        show_polygon = st.checkbox("Показать кадастровый полигон", value=True)
        
        st.subheader("Информация")
        st.info("""
        **Цвета на изображении:**
        - Красный: Маска сегментации
        - Зеленый: Контур (нейросеть)
        - Синий: Кадастровый полигон
        """)
    
    # Основной интерфейс
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Ввод данных")
        
        # Поле ввода кадастрового номера
        cadastral_number = st.text_input(
            "Кадастровый номер участка",
            placeholder="Введите номер",
            help="Формат: XX:XX:XXXXXX:XXXX"
        )
        
        # Валидация номера
        if cadastral_number:
            if validate_cadastral_number(cadastral_number):
                process_button_disabled = False
            else:
                st.error("Некорректный формат кадастрового номера!")
                st.info("Правильный формат: XX:XX:XXXXXX:XXXX (например, 38:06:144003:4723)")
                process_button_disabled = True
        else:
            process_button_disabled = True
        
        # Кнопка обработки
        process_button = st.button(
            "Проверить участок",
            disabled=process_button_disabled,
            type="primary"
        )
    
    with col2:
        st.header("Результаты")
        
        if process_button and cadastral_number:
            with st.spinner('Загрузка данных участка...'):
                # Обработка участка
                result = st.session_state.processor.process_with_mask_and_polygon(
                    cadastral_number,
                    show_mask=show_mask,
                    show_contour=show_contour,
                    show_polygon=show_polygon
                )
                
                if not result['success']:
                    st.error(f"Ошибка: {result['error']}")
                else:
                    # Анализ метрик
                    with st.spinner('Вычисление метрик...'):
                        metrics = st.session_state.processor.analyze_mask_polygon_overlap(cadastral_number)
                    
                    if metrics['success']:
                        # Отображение результатов
                        display_results(result, metrics, cadastral_number)
                    else:
                        st.error(f"Ошибка при вычислении метрик: {metrics['error']}")
                    
                    # Очистка временных файлов
                    st.session_state.processor.cleanup_files([
                        result.get('original_filename', ''),
                        result.get('result_filename', '')
                    ])


def display_results(result, metrics, cadastral_number):
    
    # Информация об участке
    st.subheader(f"Участок {cadastral_number}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Широта", f"{result['center_coords'][1]:.6f}")
        st.metric("Долгота", f"{result['center_coords'][0]:.6f}")
    
    with col2:
        st.metric("Точек полигона", len(result['coordinates']))
    
    # Отображение изображений
    st.subheader("Результат анализа")
    
    # Две колонки для изображений
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.write("**Исходное изображение**")
        st.image(result['original_image'], use_container_width=True)
    
    with img_col2:
        st.write("**Результат анализа**")
        st.image(result['result_image'], use_container_width=True)
    
    # Метрики точности
    st.subheader("Метрики точности")
    
    # Создаем метрики в колонках
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            "IoU (Пересечение)", 
            f"{metrics['iou']:.3f}",
            help="Intersection over Union - мера совпадения областей"
        )
    
    with metric_col2:
        st.metric(
            "Точность (Precision)", 
            f"{metrics['precision']:.3f}",
            help="Доля правильно предсказанных пикселей"
        )
    
    with metric_col3:
        st.metric(
            "Полнота (Recall)", 
            f"{metrics['recall']:.3f}",
            help="Доля найденных пикселей участка"
        )
    
    # Дополнительная информация
    st.subheader("Детальная информация")
    
    # Создаем DataFrame с метриками
    metrics_data = {
        'Метрика': ['Площадь маски (пикс.)', 'Площадь полигона (пикс.)', 'Площадь пересечения (пикс.)'],
        'Значение': [metrics['mask_area'], metrics['polygon_area'], metrics['intersection_area']]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    # Конвертируем изображение в байты для загрузки
    import io
    img_buffer = io.BytesIO()
    result['result_image'].save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    st.download_button(
        label="Скачать отчет",
        data=img_bytes,
        file_name=f"cadastral_{cadastral_number.replace(':', '_')}_analysis.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()