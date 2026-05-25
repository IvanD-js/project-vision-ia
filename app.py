from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import threading
import time
import av
import os

from deepface import DeepFace
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer



# ==========================================================
# CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ==========================================================

st.set_page_config(
    page_title="VisionIA | Detección de objetos y expresiones",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

CARPETA_RESULTADOS = BASE_DIR / "resultados"
CARPETA_MODELOS = BASE_DIR / "modelos"

CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_MODELOS.mkdir(exist_ok=True)

RUTA_MODELO_YOLO = BASE_DIR / "yolov8n.pt"
RUTA_MODELO_MEDIAPIPE = CARPETA_MODELOS / "blaze_face_short_range.tflite"

URL_MODELO_MEDIAPIPE = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

GUARDAR_EVIDENCIAS_EN_SERVIDOR = (
    os.getenv("VISIONIA_GUARDAR_EVIDENCIAS", "false").lower() == "true"
)


# ==========================================================
# ESTILOS PERSONALIZADOS DE LA INTERFAZ
# ==========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }

        .info-card {
            background-color: #FFFFFF;
            padding: 1.1rem;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            margin-bottom: 1rem;
        }

        .success-box {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 0.9rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }

        .warning-box {
            background-color: #FFFBEB;
            color: #92400E;
            border: 1px solid #FDE68A;
            padding: 0.9rem;
            border-radius: 12px;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# FUNCIONES PARA CARGAR LOS MODELOS
# ==========================================================

def asegurar_modelo_mediapipe() -> None:
    """
    Verifica que exista el modelo de detección facial.
    Si no existe, lo descarga automáticamente.
    """

    if not RUTA_MODELO_MEDIAPIPE.exists():
        urlretrieve(URL_MODELO_MEDIAPIPE, RUTA_MODELO_MEDIAPIPE)


@st.cache_resource(show_spinner=False)
def cargar_modelo_yolo():
    """
    Carga el modelo YOLO una sola vez.
    Streamlit vuelve a ejecutar el archivo cuando el usuario interactúa
    con la interfaz, por eso se utiliza cache_resource.
    """

    if RUTA_MODELO_YOLO.exists():
        return YOLO(str(RUTA_MODELO_YOLO))

    return YOLO("yolov8n.pt")


# ==========================================================
# FUNCIONES DE TRADUCCIÓN
# ==========================================================

# ==========================================================
# EXPRESIONES PERMITIDAS POR EL ALCANCE DEL PROYECTO
# ==========================================================

EMOCIONES_OBJETIVO = {
    "happy": "Alegría",
    "sad": "Tristeza",
    "angry": "Enojo"
}

TEXTO_SIN_OBJETIVO = "Sin expresión objetivo"


def interpretar_expresion_proyecto(analisis: dict) -> tuple[str, float]:
    """
    Interpreta la salida de DeepFace respetando el alcance del proyecto.

    El modelo puede producir categorías adicionales, pero la aplicación
    solamente muestra Alegría, Tristeza o Enojo. Si la predicción dominante
    corresponde a otra categoría, se informa que no existe una expresión
    objetivo detectada.
    """

    emocion_dominante = str(
        analisis.get("dominant_emotion", "")
    ).lower()

    puntajes = analisis.get("emotion", {})

    if emocion_dominante in EMOCIONES_OBJETIVO:
        expresion = EMOCIONES_OBJETIVO[emocion_dominante]
        confianza = float(puntajes.get(emocion_dominante, 0.0))

        return expresion, confianza

    return TEXTO_SIN_OBJETIVO, 0.0


def traducir_objeto(objeto: str) -> str:
    """
    Traduce algunas clases comunes detectadas por YOLO.
    Cuando una clase no está incluida, conserva su nombre original.
    """

    objetos = {
        "person": "Persona",
        "tie": "Corbata",
        "cell phone": "Celular",
        "laptop": "Computadora portátil",
        "keyboard": "Teclado",
        "mouse": "Ratón",
        "chair": "Silla",
        "bottle": "Botella",
        "cup": "Taza",
        "backpack": "Mochila",
        "dog": "Perro", 
        "cat": "Gato",
        "car": "Automóvil",
        "book": "Libro",
        "tv": "Pantalla",
        "clock": "Reloj",
        "remote": "Control remoto",
        "dining table": "Mesa",
        "potted plant": "Planta",
        "handbag": "Bolsa",
        "scissors": "Tijeras"
    }
        
    return objetos.get(objeto.lower(), objeto.capitalize())


def texto_para_imagen(texto: str) -> str:
    """
    Elimina acentos solamente para dibujar texto con OpenCV,
    debido a que putText puede mostrar incorrectamente caracteres especiales.
    """

    traduccion = str.maketrans(
        "áéíóúÁÉÍÓÚñÑ",
        "aeiouAEIOUnN"
    )

    return texto.translate(traduccion)

def etiqueta_expresion_para_imagen(
    expresion: str,
    confianza: float
) -> str:
    """
    Construye la etiqueta que se mostrará sobre el rostro.
    No se presenta porcentaje cuando la predicción queda fuera
    de las tres expresiones requeridas por el proyecto.
    """

    if expresion == TEXTO_SIN_OBJETIVO:
        return "Expresion: Sin expresion objetivo"

    return f"Expresion: {expresion} {confianza:.1f}%"


# ==========================================================
# FUNCIONES PARA PROCESAR Y MOSTRAR IMÁGENES
# ==========================================================

def convertir_archivo_a_imagen(archivo) -> np.ndarray:
    """
    Convierte la imagen subida o capturada desde Streamlit
    a una imagen que OpenCV pueda procesar.
    """

    bytes_imagen = archivo.getvalue()
    arreglo = np.frombuffer(bytes_imagen, np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError("No fue posible leer la imagen seleccionada.")

    return imagen


def dibujar_etiqueta(
    imagen: np.ndarray,
    texto: str,
    x: int,
    y: int,
    color: tuple
) -> None:
    """
    Dibuja una etiqueta con fondo de color para que el texto sea legible.
    """

    texto = texto_para_imagen(texto)

    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.58
    grosor = 2

    (ancho_texto, alto_texto), linea_base = cv2.getTextSize(
        texto,
        fuente,
        escala,
        grosor
    )

    y_texto = y - 8

    if y_texto - alto_texto - linea_base < 0:
        y_texto = y + alto_texto + 12

    cv2.rectangle(
        imagen,
        (x, y_texto - alto_texto - 8),
        (x + ancho_texto + 10, y_texto + linea_base + 3),
        color,
        -1
    )

    cv2.putText(
        imagen,
        texto,
        (x + 5, y_texto - 3),
        fuente,
        escala,
        (255, 255, 255),
        grosor
    )


def detectar_objetos(
    imagen_original: np.ndarray,
    imagen_resultado: np.ndarray,
    confianza_minima: float
) -> list:
    """
    Detecta objetos generales utilizando YOLO.
    Dibuja los cuadros sobre la imagen resultado.
    """

    modelo_yolo = cargar_modelo_yolo()

    prediccion = modelo_yolo.predict(
        source=imagen_original,
        conf=confianza_minima,
        verbose=False
    )[0]

    objetos_detectados = []

    if prediccion.boxes is None:
        return objetos_detectados

    for caja in prediccion.boxes:
        coordenadas = caja.xyxy[0].cpu().numpy().astype(int)

        x_inicial, y_inicial, x_final, y_final = coordenadas

        clase_id = int(caja.cls[0])
        confianza = float(caja.conf[0])

        nombre_original = modelo_yolo.names[clase_id]
        nombre_espanol = traducir_objeto(nombre_original)

        objetos_detectados.append(
            {
                "Objeto": nombre_espanol,
                "Confianza": round(confianza * 100, 2)
            }
        )

        color_objeto = (255, 100, 20)

        cv2.rectangle(
            imagen_resultado,
            (x_inicial, y_inicial),
            (x_final, y_final),
            color_objeto,
            3
        )

        dibujar_etiqueta(
            imagen_resultado,
            f"{nombre_espanol} {confianza * 100:.1f}%",
            x_inicial,
            y_inicial,
            color_objeto
        )

    return objetos_detectados


def detectar_rostros_y_expresiones(
    imagen_original: np.ndarray,
    imagen_resultado: np.ndarray
) -> list:
    """
    Localiza rostros con MediaPipe y estima la expresión facial
    de cada rostro utilizando DeepFace.
    """

    asegurar_modelo_mediapipe()

    imagen_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

    imagen_mp = mp.Image(
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

    rostros_detectados = []

    with mp.tasks.vision.FaceDetector.create_from_options(opciones) as detector:
        resultado = detector.detect(imagen_mp)

    alto_imagen, ancho_imagen = imagen_original.shape[:2]

    for indice, deteccion in enumerate(resultado.detections, start=1):
        caja = deteccion.bounding_box

        x_inicial = max(0, int(caja.origin_x))
        y_inicial = max(0, int(caja.origin_y))

        x_final = min(ancho_imagen, x_inicial + int(caja.width))
        y_final = min(alto_imagen, y_inicial + int(caja.height))

        confianza_rostro = 0.0

        if deteccion.categories:
            confianza_rostro = float(deteccion.categories[0].score) * 100

        rostro_recortado = imagen_original[
            y_inicial:y_final,
            x_inicial:x_final
        ]

        expresion = "No identificada"
        confianza_expresion = 0.0

        if rostro_recortado.size > 0:
            try:
                analisis = DeepFace.analyze(
                    img_path=rostro_recortado,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(analisis, list):
                    analisis = analisis[0]

                expresion, confianza_expresion = interpretar_expresion_proyecto(
                   analisis
                )

            except Exception:
                expresion = "No identificada"
                confianza_expresion = 0.0

        rostros_detectados.append(
            {
                "Rostro": indice,
                "Confianza de detección": round(confianza_rostro, 2),
                "Expresión estimada": expresion,
                "Confianza de expresión": round(confianza_expresion, 2)
            }
        )

        color_rostro = (40, 180, 40)

        cv2.rectangle(
            imagen_resultado,
            (x_inicial, y_inicial),
            (x_final, y_final),
            color_rostro,
            3
        )

        dibujar_etiqueta(
            imagen_resultado,
            etiqueta_expresion_para_imagen(
                expresion,
                confianza_expresion
            ),
                x_inicial,
                y_inicial,
                color_rostro
        )

    return rostros_detectados


def generar_reporte(
    nombre_imagen: str,
    objetos_detectados: list,
    rostros_detectados: list,
    confianza_minima: float
) -> str:
    """
    Genera un reporte en texto con los resultados del análisis.
    """

    lineas = []

    lineas.append("SISTEMA DE DETECCIÓN DE OBJETOS Y EXPRESIONES FACIALES\n")
    lineas.append("=" * 65 + "\n\n")

    lineas.append(f"Imagen analizada: {nombre_imagen}\n")
    lineas.append(
        f"Fecha de análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    )
    lineas.append(
        f"Confianza mínima utilizada para YOLO: {confianza_minima * 100:.0f}%\n\n"
    )

    lineas.append("1. OBJETOS DETECTADOS CON YOLO\n")
    lineas.append("-" * 45 + "\n")

    if objetos_detectados:
        for numero, objeto in enumerate(objetos_detectados, start=1):
            lineas.append(
                f"Objeto {numero}: {objeto['Objeto']} | "
                f"Confianza: {objeto['Confianza']:.2f}%\n"
            )
    else:
        lineas.append("No se detectaron objetos.\n")

    lineas.append("\n2. ROSTROS Y EXPRESIONES FACIALES ESTIMADAS\n")
    lineas.append("-" * 45 + "\n")

    if rostros_detectados:
        for rostro in rostros_detectados:
            lineas.append(
                f"Rostro {rostro['Rostro']}: "
                f"Confianza de detección: "
                f"{rostro['Confianza de detección']:.2f}% | "
                f"Expresión estimada: "
                f"{rostro['Expresión estimada']} | "
                f"Confianza de expresión: "
                f"{rostro['Confianza de expresión']:.2f}%\n"
            )
    else:
        lineas.append("No se detectaron rostros.\n")

        lineas.append("\n3. ALCANCE DE LA CLASIFICACIÓN DE EXPRESIONES\n")
        lineas.append("-" * 45 + "\n")
        lineas.append(
            "Las expresiones evaluadas por este proyecto son únicamente: "
            "Alegría, Tristeza y Enojo. Cuando el modelo produce una "
            "categoría diferente, la aplicación muestra el mensaje "
            "'Sin expresión objetivo', evitando reportar categorías "
            "fuera del alcance definido.\n\n"
        )

        lineas.append("4. OBSERVACIÓN ÉTICA\n")
        lineas.append("-" * 45 + "\n")
        lineas.append(
            "La expresión mostrada es una estimación realizada por un "
            "modelo de inteligencia artificial a partir de rasgos visibles "
            "del rostro. No representa con certeza el estado emocional real "
            "de una persona.\n"
        )

    return "".join(lineas)


def guardar_resultados(
    imagen_resultado: np.ndarray,
    reporte: str
) -> tuple:
    """
    Guarda automáticamente la imagen procesada y el reporte TXT.
    También prepara los datos para descargarlos desde la aplicación.
    """

    fecha_archivo = datetime.now().strftime("%Y%m%d_%H%M%S")

    nombre_imagen = f"resultado_web_{fecha_archivo}.jpg"
    nombre_reporte = f"reporte_web_{fecha_archivo}.txt"

    ruta_imagen = CARPETA_RESULTADOS / nombre_imagen
    ruta_reporte = CARPETA_RESULTADOS / nombre_reporte

    if GUARDAR_EVIDENCIAS_EN_SERVIDOR:
        cv2.imwrite(str(ruta_imagen), imagen_resultado)
        ruta_reporte.write_text(reporte, encoding="utf-8")

    conversion_correcta, buffer_imagen = cv2.imencode(
        ".jpg",
        imagen_resultado
    )

    if not conversion_correcta:
        raise ValueError("No fue posible preparar la imagen para descargar.")

    return (
        nombre_imagen,
        nombre_reporte,
        buffer_imagen.tobytes(),
        reporte.encode("utf-8")
    )


def procesar_imagen(
    imagen_original: np.ndarray,
    nombre_imagen: str,
    confianza_minima: float
) -> dict:
    """
    Ejecuta todo el flujo de inteligencia artificial.
    """

    imagen_resultado = imagen_original.copy()

    objetos_detectados = detectar_objetos(
        imagen_original,
        imagen_resultado,
        confianza_minima
    )

    rostros_detectados = detectar_rostros_y_expresiones(
        imagen_original,
        imagen_resultado
    )

    reporte = generar_reporte(
        nombre_imagen,
        objetos_detectados,
        rostros_detectados,
        confianza_minima
    )

    (
        nombre_imagen_resultado,
        nombre_reporte,
        bytes_imagen,
        bytes_reporte
    ) = guardar_resultados(imagen_resultado, reporte)

    return {
        "imagen_original": imagen_original,
        "imagen_resultado": imagen_resultado,
        "objetos": objetos_detectados,
        "rostros": rostros_detectados,
        "reporte": reporte,
        "nombre_imagen_resultado": nombre_imagen_resultado,
        "nombre_reporte": nombre_reporte,
        "bytes_imagen": bytes_imagen,
        "bytes_reporte": bytes_reporte
    }
    
# ==========================================================
# PROCESAMIENTO DE CÁMARA EN TIEMPO REAL
# ==========================================================

class ProcesadorTiempoReal(VideoProcessorBase):
    """
    Procesa los fotogramas recibidos desde la cámara del navegador.

    Estrategia de rendimiento:
    - El video continúa mostrándose de forma constante.
    - YOLO se ejecuta cada cierto intervalo de tiempo.
    - MediaPipe y DeepFace se ejecutan con menor frecuencia,
      porque el análisis de expresión requiere más procesamiento.
    - Los últimos resultados detectados se dibujan sobre los
      fotogramas intermedios.
    """

    def __init__(
        self,
        modelo_yolo,
        confianza_minima: float,
        intervalo_yolo: float,
        intervalo_expresion: float
    ):
        asegurar_modelo_mediapipe()

        self.modelo_yolo = modelo_yolo
        self.confianza_minima = confianza_minima
        self.intervalo_yolo = intervalo_yolo
        self.intervalo_expresion = intervalo_expresion

        self.ultimo_analisis_yolo = 0.0
        self.ultimo_analisis_expresion = 0.0

        self.objetos_actuales = []
        self.rostros_actuales = []

        self.historial_emociones = {}
        self.ultimo_fotograma = None

        self.lock = threading.Lock()

        self.ultimo_instante_frame = time.perf_counter()
        self.fps_promedio = 0.0
        self.timestamp_mediapipe = 0

        opciones_detector = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(RUTA_MODELO_MEDIAPIPE)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=0.5
        )

        self.detector_rostros = (
            mp.tasks.vision.FaceDetector.create_from_options(
                opciones_detector
            )
        )

    def _detectar_objetos(self, imagen: np.ndarray) -> list:
        """
        Ejecuta YOLO sobre un fotograma y devuelve objetos,
        coordenadas y confianza.
        """

        prediccion = self.modelo_yolo.predict(
            source=imagen,
            conf=self.confianza_minima,
            verbose=False
        )[0]

        objetos = []

        if prediccion.boxes is None:
            return objetos

        for caja in prediccion.boxes:
            coordenadas = caja.xyxy[0].cpu().numpy().astype(int)

            x_inicial, y_inicial, x_final, y_final = coordenadas

            clase_id = int(caja.cls[0])
            confianza = float(caja.conf[0]) * 100

            nombre_original = self.modelo_yolo.names[clase_id]
            nombre_espanol = traducir_objeto(nombre_original)

            objetos.append(
                {
                    "Objeto": nombre_espanol,
                    "Confianza": round(confianza, 2),
                    "Coordenadas": (
                        int(x_inicial),
                        int(y_inicial),
                        int(x_final),
                        int(y_final)
                    )
                }
            )

        return objetos

    def _suavizar_expresion(
        self,
        indice_rostro: int,
        expresion: str,
        confianza: float
    ) -> tuple:
        """
        Conserva las tres predicciones recientes de cada rostro.
        Esto evita que la etiqueta cambie demasiado rápido entre
        alegría, neutralidad o enojo debido a variaciones pequeñas.
        """

        if indice_rostro not in self.historial_emociones:
            self.historial_emociones[indice_rostro] = deque(maxlen=3)

        historial = self.historial_emociones[indice_rostro]

        historial.append((expresion, confianza))

        expresiones = [elemento[0] for elemento in historial]

        expresion_frecuente = Counter(expresiones).most_common(1)[0][0]

        confianzas_misma_expresion = [
            elemento[1]
            for elemento in historial
            if elemento[0] == expresion_frecuente
        ]

        confianza_promedio = (
            sum(confianzas_misma_expresion)
            / len(confianzas_misma_expresion)
        )

        return expresion_frecuente, confianza_promedio

    def _detectar_rostros_y_expresiones(self, imagen: np.ndarray) -> list:
        """
        Localiza rostros mediante MediaPipe y estima su expresión
        utilizando DeepFace.
        """

        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imagen_rgb
        )

        tiempo_actual_ms = int(time.perf_counter() * 1000)

        self.timestamp_mediapipe = max(
            self.timestamp_mediapipe + 1,
            tiempo_actual_ms
        )

        resultado = self.detector_rostros.detect_for_video(
            imagen_mp,
            self.timestamp_mediapipe
        )

        alto_imagen, ancho_imagen = imagen.shape[:2]

        rostros = []

        for indice, deteccion in enumerate(resultado.detections, start=1):
            caja = deteccion.bounding_box

            x_inicial = max(0, int(caja.origin_x))
            y_inicial = max(0, int(caja.origin_y))

            x_final = min(
                ancho_imagen,
                x_inicial + int(caja.width)
            )

            y_final = min(
                alto_imagen,
                y_inicial + int(caja.height)
            )

            confianza_rostro = 0.0

            if deteccion.categories:
                confianza_rostro = (
                    float(deteccion.categories[0].score) * 100
                )

            rostro_recortado = imagen[
                y_inicial:y_final,
                x_inicial:x_final
            ]

            expresion = "No identificada"
            confianza_expresion = 0.0

            if rostro_recortado.size > 0:
                try:
                    analisis = DeepFace.analyze(
                        img_path=rostro_recortado,
                        actions=["emotion"],
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(analisis, list):
                        analisis = analisis[0]

                    expresion, confianza_expresion = interpretar_expresion_proyecto(
                        analisis
                    )

                    if expresion == TEXTO_SIN_OBJETIVO:
                        self.historial_emociones.pop(indice, None)
                    else:
                        (
                        expresion,
                        confianza_expresion
                        ) = self._suavizar_expresion(
                            indice,
                            expresion,
                            confianza_expresion
                        )

                except Exception:
                    expresion = "No identificada"
                    confianza_expresion = 0.0

            rostros.append(
                {
                    "Rostro": indice,
                    "Confianza de detección": round(
                        confianza_rostro,
                        2
                    ),
                    "Expresión estimada": expresion,
                    "Confianza de expresión": round(
                        confianza_expresion,
                        2
                    ),
                    "Coordenadas": (
                        x_inicial,
                        y_inicial,
                        x_final,
                        y_final
                    )
                }
            )

        return rostros

    def _dibujar_resultados(self, imagen: np.ndarray) -> np.ndarray:
        """
        Dibuja los últimos objetos y rostros encontrados sobre
        el fotograma actual.
        """

        imagen_resultado = imagen.copy()

        color_objeto = (255, 100, 20)
        color_rostro = (40, 180, 40)

        for objeto in self.objetos_actuales:
            (
                x_inicial,
                y_inicial,
                x_final,
                y_final
            ) = objeto["Coordenadas"]

            cv2.rectangle(
                imagen_resultado,
                (x_inicial, y_inicial),
                (x_final, y_final),
                color_objeto,
                2
            )

            dibujar_etiqueta(
                imagen_resultado,
                (
                    f"{objeto['Objeto']} "
                    f"{objeto['Confianza']:.1f}%"
                ),
                x_inicial,
                y_inicial,
                color_objeto
            )

        for rostro in self.rostros_actuales:
            (
                x_inicial,
                y_inicial,
                x_final,
                y_final
            ) = rostro["Coordenadas"]

            cv2.rectangle(
                imagen_resultado,
                (x_inicial, y_inicial),
                (x_final, y_final),
                color_rostro,
                2
            )

            dibujar_etiqueta(
                imagen_resultado,
                etiqueta_expresion_para_imagen(
                    rostro["Expresión estimada"],
                    rostro["Confianza de expresión"]
                ),
                x_inicial,
                y_inicial,
                color_rostro
            )

        cv2.putText(
            imagen_resultado,
            f"FPS aproximados: {self.fps_promedio:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            3
        )

        cv2.putText(
            imagen_resultado,
            f"FPS aproximados: {self.fps_promedio:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1
        )

        return imagen_resultado

    def obtener_ultimo_resultado(self):
        """
        Permite recuperar de forma segura el último fotograma
        procesado cuando el usuario desea guardarlo.
        """

        with self.lock:
            if self.ultimo_fotograma is None:
                return None

            return {
                "imagen": self.ultimo_fotograma.copy(),
                "objetos": [
                    elemento.copy()
                    for elemento in self.objetos_actuales
                ],
                "rostros": [
                    elemento.copy()
                    for elemento in self.rostros_actuales
                ]
            }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Función ejecutada continuamente por streamlit-webrtc.
        Recibe un fotograma, aplica el análisis necesario y
        devuelve el fotograma procesado.
        """

        imagen = frame.to_ndarray(format="bgr24")

        instante_actual = time.perf_counter()

        diferencia_tiempo = (
            instante_actual - self.ultimo_instante_frame
        )

        if diferencia_tiempo > 0:
            fps_actual = 1 / diferencia_tiempo

            if self.fps_promedio == 0:
                self.fps_promedio = fps_actual
            else:
                self.fps_promedio = (
                    self.fps_promedio * 0.90
                    + fps_actual * 0.10
                )

        self.ultimo_instante_frame = instante_actual

        if (
            instante_actual - self.ultimo_analisis_yolo
            >= self.intervalo_yolo
        ):
            self.objetos_actuales = self._detectar_objetos(imagen)
            self.ultimo_analisis_yolo = instante_actual

        if (
            instante_actual - self.ultimo_analisis_expresion
            >= self.intervalo_expresion
        ):
            self.rostros_actuales = (
                self._detectar_rostros_y_expresiones(imagen)
            )

            self.ultimo_analisis_expresion = instante_actual

        imagen_resultado = self._dibujar_resultados(imagen)

        with self.lock:
            self.ultimo_fotograma = imagen_resultado.copy()

        return av.VideoFrame.from_ndarray(
            imagen_resultado,
            format="bgr24"
        )


def mostrar_camara_tiempo_real(confianza_minima: float) -> None:
    """
    Construye la sección visual para procesar video continuo
    desde la cámara del navegador.
    """

    st.divider()
    st.header("Detección con cámara en tiempo real")

    st.markdown(
        """
        La cámara transmitirá video continuamente. Los cuadros azules
        corresponden a objetos detectados con YOLO y los cuadros verdes
        corresponden a rostros con una expresión facial estimada.
        """
    )

    columna_frecuencia_yolo, columna_frecuencia_expresion = st.columns(2)

    with columna_frecuencia_yolo:
        intervalo_yolo = st.select_slider(
            "Frecuencia de detección de objetos:",
            options=[0.15, 0.25, 0.50, 1.00],
            value=0.25,
            format_func=lambda valor: f"Cada {valor:.2f} segundos"
        )

    with columna_frecuencia_expresion:
        intervalo_expresion = st.select_slider(
            "Frecuencia de análisis de expresión:",
            options=[0.50, 1.00, 1.50, 2.00],
            value=1.00,
            format_func=lambda valor: f"Cada {valor:.2f} segundos"
        )

    st.caption(
        "Para aplicar cambios en estos controles mientras el video está "
        "activo, detén la cámara y vuelve a iniciarla."
    )

    modelo_yolo_tiempo_real = cargar_modelo_yolo()

    contexto_video = webrtc_streamer(
        key="visionia-camara-tiempo-real",
        video_processor_factory=lambda: ProcesadorTiempoReal(
            modelo_yolo=modelo_yolo_tiempo_real,
            confianza_minima=confianza_minima,
            intervalo_yolo=intervalo_yolo,
            intervalo_expresion=intervalo_expresion
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=False
    )

    if contexto_video.state.playing:
        st.success(
            "Cámara activa. El sistema está procesando video en tiempo real."
        )
    else:
        st.info(
            "Presiona START y permite el acceso a la cámara para comenzar."
        )

    st.subheader("Guardar evidencia del video")

    st.write(
        "Cuando tengas una detección visible en pantalla, guarda el "
        "fotograma actual para conservarlo como evidencia del proyecto."
    )

    boton_guardar_fotograma = st.button(
        "💾 Guardar fotograma actual",
        type="primary",
        use_container_width=True,
        disabled=not contexto_video.state.playing
    )

    if boton_guardar_fotograma:
        procesador = contexto_video.video_processor

        if procesador is None:
            st.warning(
                "Todavía no existe un fotograma procesado. "
                "Espera unos segundos y vuelve a intentarlo."
            )
        else:
            captura = procesador.obtener_ultimo_resultado()

            if captura is None:
                st.warning(
                    "Todavía no existe un fotograma procesado. "
                    "Espera unos segundos y vuelve a intentarlo."
                )
            else:
                reporte = generar_reporte(
                    "captura_tiempo_real.jpg",
                    captura["objetos"],
                    captura["rostros"],
                    confianza_minima
                )

                (
                    nombre_imagen,
                    nombre_reporte,
                    bytes_imagen,
                    bytes_reporte
                ) = guardar_resultados(
                    captura["imagen"],
                    reporte
                )

                st.session_state["captura_tiempo_real"] = {
                    "imagen": captura["imagen"],
                    "nombre_imagen": nombre_imagen,
                    "nombre_reporte": nombre_reporte,
                    "bytes_imagen": bytes_imagen,
                    "bytes_reporte": bytes_reporte,
                    "reporte": reporte
                }

    if "captura_tiempo_real" in st.session_state:
        evidencia = st.session_state["captura_tiempo_real"]

        st.success(
            "Fotograma guardado correctamente en la carpeta resultados."
        )

        imagen_evidencia_rgb = cv2.cvtColor(
            evidencia["imagen"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            imagen_evidencia_rgb,
            caption="Último fotograma guardado",
            use_container_width=True
        )

        columna_descarga_foto, columna_descarga_txt = st.columns(2)

        with columna_descarga_foto:
            st.download_button(
                label="⬇️ Descargar fotograma procesado",
                data=evidencia["bytes_imagen"],
                file_name=evidencia["nombre_imagen"],
                mime="image/jpeg",
                use_container_width=True
            )

        with columna_descarga_txt:
            st.download_button(
                label="📄 Descargar reporte del fotograma",
                data=evidencia["bytes_reporte"],
                file_name=evidencia["nombre_reporte"],
                mime="text/plain",
                use_container_width=True
            )


# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================

st.markdown(
    '<div class="main-title">🤖 VisionIA</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Sistema de detección de objetos y estimación de expresiones faciales
        mediante inteligencia artificial.
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Configuración")

    modo_entrada = st.radio(
    "Selecciona el origen de la imagen:",
    [
        "Subir imagen",
        "Tomar fotografía",
        "Cámara en tiempo real"
    ]
    )

    confianza_minima = st.slider(
        "Confianza mínima para objetos:",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Los objetos con confianza menor a este valor no serán mostrados."
    )

    st.divider()

    st.markdown("### Tecnologías utilizadas")

    st.markdown(
        """
        - **YOLO:** detección de objetos.
        - **MediaPipe:** detección de rostros.
        - **DeepFace:** estimación de expresiones.
        - **Streamlit:** interfaz web.
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            El sistema estima expresiones faciales visibles,
            pero no determina con certeza las emociones reales
            de una persona.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.info(
    "Las imágenes se procesan únicamente para mostrar el resultado. "
    "En la versión pública no se almacenan automáticamente en el servidor."
)
    


columna_entrada, columna_informacion = st.columns([1.4, 1])

with columna_entrada:
    st.subheader("Entrada de imagen")

    archivo_entrada = None

    if modo_entrada == "Subir imagen":
        archivo_entrada = st.file_uploader(
            "Selecciona una fotografía",
            type=["jpg", "jpeg", "png"],
            help="Puedes cargar imágenes JPG, JPEG o PNG."
        )

    elif modo_entrada == "Tomar fotografía":
        archivo_entrada = st.camera_input(
            "Toma una fotografía para analizarla"
        )

    else:
        st.info(
            "La transmisión continua aparecerá debajo de esta sección. "
            "Presiona START para activar la webcam."
        )

with columna_informacion:
    st.subheader("Objetivo del sistema")

    st.markdown(
        """
        <div class="info-card">
            Esta aplicación procesa fotografías para localizar objetos
            presentes en la escena y detectar rostros humanos. Cuando
            se encuentra un rostro, el sistema estima si su expresión
            corresponde a alegría, tristeza, enojo u otra categoría
            disponible en el modelo.
        </div>
        """,
        unsafe_allow_html=True
    )
    
if modo_entrada == "Cámara en tiempo real":
    mostrar_camara_tiempo_real(confianza_minima)
    st.stop()


if archivo_entrada is not None:
    try:
        imagen_para_analizar = convertir_archivo_a_imagen(archivo_entrada)

        nombre_archivo = getattr(
            archivo_entrada,
            "name",
            "captura_camara.jpg"
        )

        st.divider()

        columna_previa, columna_boton = st.columns([1.5, 1])

        with columna_previa:
            st.subheader("Vista previa")

            imagen_rgb = cv2.cvtColor(
                imagen_para_analizar,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                imagen_rgb,
                caption="Imagen seleccionada para análisis",
                use_container_width=True
            )

        with columna_boton:
            st.subheader("Procesamiento")

            st.write(
                "Presiona el botón para ejecutar la detección de objetos "
                "y el análisis de expresiones faciales."
            )

            boton_analizar = st.button(
                "🔍 Analizar imagen",
                type="primary",
                use_container_width=True
            )

        if boton_analizar:
            with st.spinner(
                "Procesando imagen con los modelos de inteligencia artificial..."
            ):
                resultado = procesar_imagen(
                    imagen_para_analizar,
                    nombre_archivo,
                    confianza_minima
                )

                st.session_state["ultimo_resultado"] = resultado

            st.markdown(
                """
                <div class="success-box">
                    Análisis completado correctamente. Los archivos también
                    fueron guardados automáticamente en la carpeta resultados.
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as error:
        st.error(f"Ocurrió un error al leer o procesar la imagen: {error}")


if "ultimo_resultado" in st.session_state:
    resultado = st.session_state["ultimo_resultado"]

    st.divider()
    st.header("Resultados del análisis")

    pestana_visual, pestana_datos, pestana_reporte = st.tabs(
        ["Resultado visual", "Detecciones", "Reporte generado"]
    )

    with pestana_visual:
        columna_original, columna_resultado = st.columns(2)

        with columna_original:
            st.subheader("Imagen original")

            original_rgb = cv2.cvtColor(
                resultado["imagen_original"],
                cv2.COLOR_BGR2RGB
            )

            st.image(
                original_rgb,
                use_container_width=True
            )

        with columna_resultado:
            st.subheader("Imagen procesada")

            resultado_rgb = cv2.cvtColor(
                resultado["imagen_resultado"],
                cv2.COLOR_BGR2RGB
            )

            st.image(
                resultado_rgb,
                use_container_width=True
            )

        columna_descarga_imagen, columna_descarga_reporte = st.columns(2)

        with columna_descarga_imagen:
            st.download_button(
                label="⬇️ Descargar imagen procesada",
                data=resultado["bytes_imagen"],
                file_name=resultado["nombre_imagen_resultado"],
                mime="image/jpeg",
                use_container_width=True
            )

        with columna_descarga_reporte:
            st.download_button(
                label="📄 Descargar reporte TXT",
                data=resultado["bytes_reporte"],
                file_name=resultado["nombre_reporte"],
                mime="text/plain",
                use_container_width=True
            )

    with pestana_datos:
        st.subheader("Objetos detectados con YOLO")

        if resultado["objetos"]:
            tabla_objetos = pd.DataFrame(resultado["objetos"])
            tabla_objetos["Confianza"] = (
                tabla_objetos["Confianza"].astype(str) + " %"
            )

            st.dataframe(
                tabla_objetos,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se detectaron objetos en la imagen.")

        st.subheader("Rostros y expresiones estimadas")

        if resultado["rostros"]:
            tabla_rostros = pd.DataFrame(resultado["rostros"])

            tabla_rostros["Confianza de detección"] = (
                tabla_rostros["Confianza de detección"].astype(str) + " %"
            )

            tabla_rostros["Confianza de expresión"] = (
                tabla_rostros["Confianza de expresión"].astype(str) + " %"
            )

            st.dataframe(
                tabla_rostros,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se detectaron rostros en la imagen.")

    with pestana_reporte:
        st.subheader("Documento de resultados")

        st.text_area(
            "Contenido del reporte:",
            resultado["reporte"],
            height=380
        )

else:
    st.info(
        "Sube una imagen o toma una fotografía para comenzar el análisis."
    )