## Описание
Это GUI-приложение для оценки глубины изображения с помощью камеры Intel Realsense D415. В репозитории доступе код для обработки сетью YOLO, но обработка не интегрирована в пайплайн из-за проблем с тестовой камерой.
## Установка
1. Установите зависимости
```shell
conda create -n intel-gui Python=3.11
conda activate intel-gui
pip install -r requirements.txt
```
2. Опционально (для ИИ-обработки): добавьте [YOLO-модели](https://github.com/ultralytics/ultralytics) в форматах `.pt` и `.onnx` в директорию `models/`. Для конвертации из `.pt` в `.onnx` можно воспользоваться методом `pt2onnx` в модуле `models.py`. 
## Запуск 
```shell
python gui.py
```