from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from deepface import DeepFace
from ultralytics import YOLO


# ==========================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

CARPETA_IMAGENES = BASE_DIR / "imagenes"
CARPETA_RESULTADOS = BASE_DIR / "resultados"
CARPETA_MODELOS = BASE_DIR / "modelos"

CARPETA_IMAGENES.mkdir(exist_ok=True)
CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_MODELOS.mkdir(exist_ok=True)

RUTA_MODELO_MEDIAPIPE = CARPETA_MODELOS / "blaze_face_short_range.tflite"

URL_MODELO_MEDIAPIPE = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


# ==========================================================
# DESCARGA DEL MODELO DE DETECCIÓN FACIAL
# ==========================================================

def descargar_modelo_mediapipe() -> None:
    """
    Descarga el modelo BlazeFace de MediaPipe solamente si aún no existe.
    Este modelo permite localizar rostros en una imagen.
    """

    if not RUTA_MODELO_MEDIAPIPE.exists():
        print("Descargando modelo facial de MediaPipe...")
        urlretrieve(URL_MODELO_MEDIAPIPE, RUTA_MODELO_MEDIAPIPE)
        print(f"Modelo guardado en: {RUTA_MODELO_MEDIAPIPE}")


descargar_modelo_mediapipe()


# ==========================================================
# CARGA DEL MODELO YOLO
# ==========================================================

# Si yolov8n.pt ya se descargó anteriormente en la carpeta del proyecto,
# lo reutilizamos. De lo contrario, Ultralytics lo descargará automáticamente.
RUTA_MODELO_YOLO = BASE_DIR / "yolov8n.pt"

if RUTA_MODELO_YOLO.exists():
    modelo_yolo = YOLO(str(RUTA_MODELO_YOLO))
else:
    modelo_yolo = YOLO("yolov8n.pt")


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def traducir_emocion(emocion: str) -> str:
    """
    Convierte las etiquetas de DeepFace del inglés al español.
    """

    traducciones = {
        "happy": "Alegría",
        "sad": "Tristeza",
        "angry": "Enojo",
        "neutral": "Neutral",
        "fear": "Miedo",
        "surprise": "Sorpresa",
        "disgust": "Disgusto"
    }

    return traducciones.get(emocion.lower(), "No identificada")


def quitar_acentos_para_imagen(texto: str) -> str:
    """
    OpenCV puede mostrar incorrectamente los acentos.
    Esta función únicamente adapta el texto que se dibuja en la imagen.
    """

    reemplazos = str.maketrans(
        "áéíóúÁÉÍÓÚñÑ",
        "aeiouAEIOUnN"
    )

    return texto.translate(reemplazos)

def redimensionar_para_mostrar(imagen, ancho_maximo=900, alto_maximo=650):
    """
    Reduce únicamente la imagen que se muestra en pantalla.
    La imagen guardada en la carpeta resultados mantiene su calidad original.
    """

    alto_original, ancho_original = imagen.shape[:2]

    escala_ancho = ancho_maximo / ancho_original
    escala_alto = alto_maximo / alto_original

    # Elegimos la escala menor para no deformar la imagen.
    # El valor 1.0 evita agrandar imágenes pequeñas.
    escala = min(escala_ancho, escala_alto, 1.0)

    nuevo_ancho = int(ancho_original * escala)
    nuevo_alto = int(alto_original * escala)

    if escala < 1.0:
        imagen_redimensionada = cv2.resize(
            imagen,
            (nuevo_ancho, nuevo_alto),
            interpolation=cv2.INTER_AREA
        )
        return imagen_redimensionada

    return imagen


def detectar_rostros_mediapipe(imagen_bgr):
    """
    Detecta rostros utilizando la API actual de MediaPipe Tasks.
    Recibe una imagen de OpenCV en formato BGR.
    Devuelve los resultados de detección facial.
    """

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    imagen_mediapipe = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imagen_rgb
    )

    opciones = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(RUTA_MODELO_MEDIAPIPE)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    with mp.tasks.vision.FaceDetector.create_from_options(opciones) as detector:
        resultado = detector.detect(imagen_mediapipe)

    return resultado


# ==========================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ==========================================================

def analizar_imagen(ruta_ingresada: str) -> None:
    """
    Ejecuta todo el procesamiento del proyecto:

    1. Carga una imagen JPG.
    2. Detecta objetos con YOLO.
    3. Detecta rostros con MediaPipe.
    4. Estima la expresión facial con DeepFace.
    5. Dibuja los resultados.
    6. Guarda una imagen y un archivo de texto como evidencia.
    """

    ruta_imagen = Path(ruta_ingresada.strip().strip('"'))

    if not ruta_imagen.is_absolute():
        ruta_imagen = BASE_DIR / ruta_imagen

    imagen_original = cv2.imread(str(ruta_imagen))

    if imagen_original is None:
        print("\nERROR: No se pudo abrir la imagen.")
        print(f"Ruta buscada: {ruta_imagen}")
        print("Verifica el nombre del archivo y que sea una imagen JPG válida.")
        return

    lineas_reporte = []

    lineas_reporte.append("SISTEMA DE DETECCIÓN DE OBJETOS Y EXPRESIONES FACIALES\n")
    lineas_reporte.append("=" * 60 + "\n")
    lineas_reporte.append(f"Imagen analizada: {ruta_imagen.name}\n")
    lineas_reporte.append(
        f"Fecha de análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
    )

    # ======================================================
    # 1. DETECCIÓN DE OBJETOS CON YOLO
    # ======================================================

    print("Procesando detección de objetos con YOLO...")

    resultado_yolo = modelo_yolo.predict(
        source=imagen_original,
        conf=0.35,
        verbose=False
    )[0]

    # YOLO devuelve una copia de la imagen con cajas de objetos dibujadas.
    imagen_resultado = resultado_yolo.plot()

    lineas_reporte.append("1. DETECCIÓN DE OBJETOS CON YOLO\n")
    lineas_reporte.append("-" * 40 + "\n")

    if resultado_yolo.boxes is not None and len(resultado_yolo.boxes) > 0:
        for indice, caja in enumerate(resultado_yolo.boxes, start=1):
            clase_id = int(caja.cls[0])
            confianza = float(caja.conf[0])
            nombre_objeto = modelo_yolo.names[clase_id]

            lineas_reporte.append(
                f"Objeto {indice}: {nombre_objeto} | "
                f"Confianza: {confianza:.2f}\n"
            )
    else:
        lineas_reporte.append("No se detectaron objetos.\n")

    # ======================================================
    # 2. DETECCIÓN DE ROSTROS CON MEDIAPIPE
    # ======================================================

    print("Procesando detección de rostros con MediaPipe...")

    resultado_rostros = detectar_rostros_mediapipe(imagen_original)

    lineas_reporte.append("\n2. DETECCIÓN DE ROSTROS Y EXPRESIONES\n")
    lineas_reporte.append("-" * 40 + "\n")

    alto, ancho = imagen_original.shape[:2]

    if resultado_rostros.detections:
        for indice, deteccion in enumerate(resultado_rostros.detections, start=1):

            caja_rostro = deteccion.bounding_box

            x = max(0, int(caja_rostro.origin_x))
            y = max(0, int(caja_rostro.origin_y))
            x_final = min(ancho, x + int(caja_rostro.width))
            y_final = min(alto, y + int(caja_rostro.height))

            confianza_rostro = 0.0

            if deteccion.categories:
                confianza_rostro = float(deteccion.categories[0].score)

            rostro_recortado = imagen_original[y:y_final, x:x_final]

            expresion = "No identificada"

            if rostro_recortado.size > 0:
                try:
                    analisis_emocion = DeepFace.analyze(
                        img_path=rostro_recortado,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(analisis_emocion, list):
                        analisis_emocion = analisis_emocion[0]

                    expresion_ingles = analisis_emocion["dominant_emotion"]
                    expresion = traducir_emocion(expresion_ingles)

                except Exception as error:
                    expresion = "No identificada"
                    lineas_reporte.append(
                        f"Advertencia rostro {indice}: "
                        f"no se pudo estimar la expresión "
                        f"({type(error).__name__}).\n"
                    )

            # Dibujar cuadro del rostro.
            cv2.rectangle(
                imagen_resultado,
                (x, y),
                (x_final, y_final),
                (0, 255, 0),
                2
            )

            etiqueta = f"Expresion: {quitar_acentos_para_imagen(expresion)}"

            # Dibujar texto sobre el rostro.
            cv2.putText(
                imagen_resultado,
                etiqueta,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            lineas_reporte.append(
                f"Rostro {indice}: detectado | "
                f"Confianza MediaPipe: {confianza_rostro:.2f} | "
                f"Expresión estimada: {expresion}\n"
            )

    else:
        lineas_reporte.append("No se detectaron rostros en la imagen.\n")

    # ======================================================
    # 3. GUARDAR RESULTADOS
    # ======================================================

    fecha_archivo = datetime.now().strftime("%Y%m%d_%H%M%S")

    ruta_resultado_imagen = (
        CARPETA_RESULTADOS / f"resultado_imagen_{fecha_archivo}.jpg"
    )

    ruta_resultado_texto = (
        CARPETA_RESULTADOS / f"resultado_reporte_{fecha_archivo}.txt"
    )

    cv2.imwrite(str(ruta_resultado_imagen), imagen_resultado)

    with open(ruta_resultado_texto, "w", encoding="utf-8") as archivo:
        archivo.writelines(lineas_reporte)

    print("\nAnálisis terminado correctamente.")
    print(f"Imagen procesada guardada en: {ruta_resultado_imagen}")
    print(f"Reporte guardado en: {ruta_resultado_texto}")

        # ======================================================
    # 4. MOSTRAR RESULTADO EN UNA VENTANA AJUSTABLE
    # ======================================================

    # Reducimos solamente la vista previa para que no ocupe toda la pantalla.
    imagen_vista_previa = redimensionar_para_mostrar(
        imagen_resultado,
        ancho_maximo=900,
        alto_maximo=650
    )

    nombre_ventana = "Resultado del analisis de IA"

    # WINDOW_NORMAL permite cambiar manualmente el tamaño de la ventana.
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    # Mostramos la imagen reducida.
    cv2.imshow(nombre_ventana, imagen_vista_previa)

    print("\nPresiona cualquier tecla sobre la ventana de la imagen para cerrarla.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# INICIO DEL PROGRAMA
# ==========================================================

if __name__ == "__main__":
    print("\nSISTEMA DE DETECCIÓN DE OBJETOS Y EXPRESIONES FACIALES")
    print("Ruta de la imagen: imagenes/persona.jpg\n")

    ruta_usuario = input("Escribe la ruta de la imagen JPG: ")

    analizar_imagen(ruta_usuario)