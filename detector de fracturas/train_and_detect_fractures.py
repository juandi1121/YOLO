import os
from ultralytics import YOLO
import cv2
from PIL import Image

data_yaml_path = 'fracture_data.yaml'

project_base_dir = 'runs/detect'
experiment_name = 'fracture_detector_yolov8'


print("\n--- Cargando el modelo YOLOv8 ---")

model = YOLO('yolov8n.pt')


print(f"\n--- Iniciando el entrenamiento del modelo '{experiment_name}' ---")
print(f"Los resultados se guardarán en: {os.path.join(project_base_dir, experiment_name)}")

results = model.train(
    data=data_yaml_path,  # Ruta a tu archivo YAML del dataset
    epochs=100,  # Número de épocas de entrenamiento (ajusta según tu dataset y recursos)
    imgsz=640,  # Tamaño de la imagen de entrada para el modelo (640x640 es estándar para YOLOv8)
    batch=16,  # Tamaño del batch (ajusta según la VRAM de tu GPU; menor si tienes poca VRAM)
    name=experiment_name,  # Nombre de esta ejecución de entrenamiento
    project=project_base_dir,  # Directorio base para guardar los resultados
    # val=True,           # Ejecutar validación después de cada época (por defecto True)
    # patience=50,        # Número de épocas sin mejora para parar el entrenamiento temprano
    # device=0,           # Usar la primera GPU (0), o 'cpu' para CPU, o una lista [0,1] para múltiples GPUs
)

print(f"\n--- Entrenamiento completado para '{experiment_name}' ---")

# La ruta al mejor modelo entrenado se puede encontrar así:
best_model_path = os.path.join(project_base_dir, experiment_name, 'weights', 'best.pt')
print(f"El modelo entrenado (best.pt) se guardó en: {best_model_path}")

# --- 4. Realizar Detecciones con el Modelo Entrenado ---

print("\n--- Cargando el modelo entrenado para realizar detecciones ---")

if not os.path.exists(best_model_path):
    print(f"Error: No se encontró el modelo entrenado en {best_model_path}.")
    print("El entrenamiento pudo haber fallado o no se completó.")
    exit()

trained_model = YOLO(best_model_path)

# --- Configura la imagen de prueba ---
# ¡IMPORTANTE: Reemplaza esta ruta con una imagen de tu dataset de validación o una imagen nueva!
# Ejemplo: '/ruta/absoluta/a/tu/dataset_fracturas/images/val/radio_val_001.jpg'
test_image_path = '/Users/quesi/Desktop/detector de fracturas/archive/BoneFractureYolo8/test/images/image1_68_png.rf.9e3dfa26e497af0a8f676a9686fd0e20.jpg'

if not os.path.exists(test_image_path):
    print(f"\nError: La imagen de prueba no se encontró en {test_image_path}.")
    print("Por favor, actualiza la variable 'test_image_path' con una ruta válida a una radiografía para probar.")
else:
    print(f"\n--- Realizando inferencia en la imagen: {test_image_path} ---")

    # El método .predict() de YOLOv8 es muy eficiente.
    # source: La imagen o carpeta de imágenes para predecir.
    # conf: Umbral de confianza. Solo se mostrarán las detecciones con una confianza superior a este valor.
    # iou: Umbral de Intersection Over Union (IoU) para Non-Maximum Suppression (NMS).
    # save: Si es True, guarda las imágenes con las detecciones en 'runs/detect/predict'.
    predictions = trained_model.predict(source=test_image_path, conf=0.5, iou=0.7, save=True)

    print("\n--- Detecciones completadas ---")

    # Los resultados se guardan automáticamente si 'save=True'.
    # Puedes acceder a los objetos de resultados para un procesamiento más detallado:
    for r in predictions:
        print(f"\nResultados para {test_image_path}:")
        print(f"Cajas delimitadoras (formato xyxy): {r.boxes.xyxy}")
        print(f"Confianzas: {r.boxes.conf}")
        print(f"Clases: {r.boxes.cls} (0 = fractura)")

        # Puedes también mostrar la imagen con las detecciones si lo deseas (requiere matplotlib o PIL)
        # im_array = r.plot()  # Obtiene la imagen numpy array con los cuadros dibujados (BGR)
        # im = Image.fromarray(im_array[..., ::-1]) # Convertir BGR a RGB para PIL
        # im.show() # Requiere que tengas un visor de imágenes predeterminado en tu sistema

    print(
        f"\nLas imágenes con las detecciones se han guardado en la carpeta '{os.path.join('runs', 'detect', 'predict')}'")
    print("Busca un archivo con el nombre de tu imagen original.")