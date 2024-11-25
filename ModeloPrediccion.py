import cv2  # Librería para capturar video y procesar imágenes.
import mediapipe as mp  # Librería para el procesamiento de poses y landmarks corporales.
import numpy as np  # Para operaciones matemáticas y manejo de arrays.
import pandas as pd  # Para estructurar y manipular datos.
import pickle  # Para cargar modelos previamente entrenados.
import tkinter as tk  # Para crear la interfaz gráfica de usuario.
from tkinter import ttk, messagebox  # Widgets adicionales para la interfaz y mensajes emergentes.
from PIL import Image, ImageTk  # Para manipular y mostrar imágenes en la interfaz.

# Clase principal para realizar predicciones en tiempo real.
class PredictionSystem:
    def __init__(self):
        """
        Inicializa el sistema de predicción, configurando Mediapipe para detección de poses,
        cargando el modelo entrenado y preparando un buffer para predicciones consecutivas.
        """
        self.mp_pose = mp.solutions.pose  # Configuración de Mediapipe para detección de poses.
        self.mp_drawing = mp.solutions.drawing_utils  # Herramientas para dibujar landmarks en los frames.
        self.pose = self.mp_pose.Pose(  # Inicializa el modelo de poses con configuraciones básicas.
            min_detection_confidence=0.5,  # Confianza mínima para detectar una pose.
            min_tracking_confidence=0.5  # Confianza mínima para el seguimiento continuo.
        )

        self.load_model()  # Carga el modelo entrenado, escalador y codificador.

        self.prediction_buffer = []  # Lista para almacenar predicciones consecutivas.
        self.buffer_size = 5  # Tamaño máximo del buffer para suavizar predicciones.

    def create_gui(self):
        """
        Crea la interfaz gráfica para mostrar el video en tiempo real y los resultados de las predicciones.
        """
        self.root = tk.Tk()  # Inicializa la ventana principal de Tkinter.
        self.root.title("Sistema de Predicción de Poses")  # Título de la ventana.

        # Frame principal para organizar los elementos de la interfaz.
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Etiqueta para mostrar el video en tiempo real.
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2)

        # Frame para mostrar los resultados de predicción.
        pred_frame = ttk.LabelFrame(main_frame, text="Predicción", padding="5")
        pred_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Etiqueta para la actividad detectada.
        self.prediction_label = ttk.Label(
            pred_frame,
            text="Actividad detectada: -",
            font=('Arial', 14, 'bold')
        )
        self.prediction_label.grid(row=0, column=0)

        # Etiqueta para mostrar la confianza de la predicción.
        self.confidence_label = ttk.Label(
            pred_frame,
            text="Confianza: -",
            font=('Arial', 12)
        )
        self.confidence_label.grid(row=1, column=0)

        # Frame para botones de control.
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=5)

        # Botón para alternar entre predicción activa e inactiva.
        ttk.Button(
            controls_frame,
            text="Iniciar/Detener",
            command=self.toggle_prediction
        ).grid(row=0, column=0, padx=5)

        self.cap = cv2.VideoCapture(0)  # Inicializa la captura de video desde la cámara.
        self.is_predicting = False  # Estado inicial del modo de predicción.

    def load_model(self):
        """
        Carga el modelo entrenado junto con el escalador y codificador.
        """
        try:
            with open('pose_model.pkl', 'rb') as f:  # Intenta cargar el archivo del modelo.
                self.model, self.scaler, self.label_encoder = pickle.load(f)
        except FileNotFoundError:  # Si el archivo no existe, muestra un error.
            messagebox.showerror(
                "Error",
                "No se encontró el modelo entrenado (pose_model.pkl)"
            )
            self.model = None  # Asigna `None` al modelo si no se carga correctamente.

    def process_frame(self, frame):
        """
        Procesa un frame del video, detectando landmarks corporales y calculando ángulos clave.
        Args:
            frame: El frame de video capturado.
        Returns:
            frame: El frame procesado con los landmarks dibujados.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte el frame a formato RGB.
        results = self.pose.process(image_rgb)  # Detecta landmarks en el frame.

        if results.pose_landmarks:  # Si se detectan landmarks:
            self.mp_drawing.draw_landmarks(  # Dibuja los landmarks en el frame.
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            landmarks_dict = self._extract_landmarks(results.pose_landmarks)  # Extrae coordenadas de landmarks.
            angles = self._calculate_key_angles(landmarks_dict)  # Calcula ángulos clave.

            if self.is_predicting:  # Si el modo de predicción está activo:
                self.predict_activity(landmarks_dict, angles)  # Predice la actividad.

        return frame  # Devuelve el frame procesado.

    def _extract_landmarks(self, landmarks):
        """
        Extrae coordenadas clave de landmarks detectados por Mediapipe.
        Args:
            landmarks: Landmarks detectados en el cuerpo.
        Returns:
            keypoints (dict): Diccionario con las coordenadas (x, y, z) de los landmarks.
        """
        keypoints = {}  # Inicializa un diccionario para almacenar los landmarks.

        # Lista de landmarks importantes y sus nombres.
        important_landmarks = {
            'LEFT_HIP': self.mp_pose.PoseLandmark.LEFT_HIP,
            'RIGHT_HIP': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'LEFT_KNEE': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'RIGHT_KNEE': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'LEFT_ANKLE': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'RIGHT_ANKLE': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'LEFT_SHOULDER': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'RIGHT_SHOULDER': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'LEFT_WRIST': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'RIGHT_WRIST': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'NOSE': self.mp_pose.PoseLandmark.NOSE
        }

        # Itera sobre cada landmark importante, extrayendo sus coordenadas.
        for name, landmark in important_landmarks.items():
            keypoints[f'{name}_x'] = landmarks.landmark[landmark].x
            keypoints[f'{name}_y'] = landmarks.landmark[landmark].y
            keypoints[f'{name}_z'] = landmarks.landmark[landmark].z

        return keypoints  # Devuelve el diccionario de landmarks.

    def _calculate_key_angles(self, landmarks):
        """
        Calcula ángulos clave a partir de las coordenadas de landmarks.
        Args:
            landmarks (dict): Diccionario con las coordenadas de los landmarks.
        Returns:
            angles (dict): Diccionario con los ángulos calculados.
        """
        angles = {}  # Diccionario para almacenar los ángulos.

        # Calcula el ángulo de la rodilla izquierda.
        left_knee_angle = self._calculate_angle(
            [landmarks['LEFT_HIP_x'], landmarks['LEFT_HIP_y']],
            [landmarks['LEFT_KNEE_x'], landmarks['LEFT_KNEE_y']],
            [landmarks['LEFT_ANKLE_x'], landmarks['LEFT_ANKLE_y']]
        )
        angles['Left Knee'] = left_knee_angle

        # Calcula el ángulo de la rodilla derecha.
        right_knee_angle = self._calculate_angle(
            [landmarks['RIGHT_HIP_x'], landmarks['RIGHT_HIP_y']],
            [landmarks['RIGHT_KNEE_x'], landmarks['RIGHT_KNEE_y']],
            [landmarks['RIGHT_ANKLE_x'], landmarks['RIGHT_ANKLE_y']]
        )
        angles['Right Knee'] = right_knee_angle

        # Calcula la inclinación del tronco.
        trunk_angle = self._calculate_angle(
            [(landmarks['LEFT_SHOULDER_x'] + landmarks['RIGHT_SHOULDER_x'])/2,
             (landmarks['LEFT_SHOULDER_y'] + landmarks['RIGHT_SHOULDER_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
             (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
            (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2 + 0.1]
        )
        angles['Trunk Inclination'] = trunk_angle  # Almacena el ángulo calculado para la inclinación del tronco.

        return angles  # Devuelve el diccionario con los ángulos calculados.

    def _calculate_angle(self, a, b, c):
        """
        Calcula el ángulo entre tres puntos utilizando sus coordenadas.
        Args:
            a, b, c: Coordenadas de los puntos en formato [x, y].
        Returns:
            angle (float): Ángulo calculado en grados.
        """
        a = np.array(a)  # Convierte el punto A a un array NumPy.
        b = np.array(b)  # Convierte el punto B a un array NumPy (punto central).
        c = np.array(c)  # Convierte el punto C a un array NumPy.

        # Calcula el ángulo en radianes utilizando funciones trigonométricas.
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)  # Convierte el ángulo a grados.

        if angle > 180.0:  # Ajusta el ángulo si supera los 180 grados.
            angle = 360-angle

        return angle  # Devuelve el ángulo calculado.

    def prepare_features(self, landmarks_dict, angles):
        """
        Prepara las características para realizar una predicción, combinando coordenadas y ángulos.
        Args:
            landmarks_dict (dict): Coordenadas de los landmarks.
            angles (dict): Ángulos calculados.
        Returns:
            features (array): Características preparadas en un array NumPy.
        """
        features = []  # Lista para almacenar las características.

        # Añade las coordenadas de los landmarks a la lista de características.
        for key in landmarks_dict.keys():
            features.append(landmarks_dict[key])

        # Añade los ángulos calculados a la lista de características.
        for key in angles.keys():
            features.append(angles[key])

        return np.array(features).reshape(1, -1)  # Convierte la lista en un array y lo reconfigura como una matriz.

    def predict_activity(self, landmarks_dict, angles):
        """
        Realiza una predicción de la actividad basada en las características extraídas.
        Args:
            landmarks_dict (dict): Coordenadas de los landmarks.
            angles (dict): Ángulos calculados.
        """
        if self.model is None:  # Si no hay modelo cargado, se detiene.
            return

        # Prepara las características para la predicción.
        features = self.prepare_features(landmarks_dict, angles)

        # Escala las características utilizando el escalador previamente entrenado.
        features_scaled = self.scaler.transform(features)

        # Realiza la predicción utilizando el modelo cargado.
        prediction_encoded = self.model.predict(features_scaled)[0]  # Predicción codificada (índice numérico).
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]  # Convierte el índice a etiqueta.

        # Obtiene las probabilidades de predicción para calcular la confianza.
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max() * 100  # Calcula la confianza como el porcentaje más alto.

        # Actualiza el buffer con la nueva predicción.
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > self.buffer_size:  # Elimina las predicciones más antiguas si el buffer está lleno.
            self.prediction_buffer.pop(0)

        # Determina la predicción más frecuente en el buffer.
        from collections import Counter
        final_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]

        # Actualiza la interfaz gráfica con la predicción y confianza.
        self.prediction_label.configure(
            text=f"Actividad detectada: {final_prediction}"
        )
        self.confidence_label.configure(
            text=f"Confianza: {confidence:.1f}%"
        )

    def update_video(self):
        """
        Actualiza el frame de video en la interfaz gráfica.
        """
        ret, frame = self.cap.read()  # Captura un frame de la cámara.
        if ret:  # Si se captura un frame válido:
            frame = self.process_frame(frame)  # Procesa el frame (detección de landmarks y predicción).

            # Convierte el frame procesado al formato RGB para mostrarlo en la GUI.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)  # Convierte el frame a una imagen PIL.
            imgtk = ImageTk.PhotoImage(image=img)  # Convierte la imagen PIL para usarla en Tkinter.
            self.video_label.imgtk = imgtk  # Asocia la imagen con el widget de video.
            self.video_label.configure(image=imgtk)  # Actualiza el widget de video.

        self.root.after(10, self.update_video)  # Programa la actualización del video cada 10 ms.

    def toggle_prediction(self):
        """
        Alterna el modo de predicción entre activo e inactivo.
        """
        self.is_predicting = not self.is_predicting  # Cambia el estado del modo de predicción.
        if not self.is_predicting:  # Si se detiene la predicción:
            self.prediction_label.configure(text="Actividad detectada: -")  # Restablece la etiqueta de actividad.
            self.confidence_label.configure(text="Confianza: -")  # Restablece la etiqueta de confianza.
            self.prediction_buffer.clear()  # Limpia el buffer de predicciones.

    def run(self):
        """
        Ejecuta la aplicación de predicción.
        """
        self.create_gui()  # Crea la interfaz gráfica.
        self.update_video()  # Inicia la actualización del video en tiempo real.
        self.root.mainloop()  # Ejecuta el bucle principal de la interfaz gráfica.

    def cleanup(self):
        """
        Libera recursos como la cámara y cierra todas las ventanas.
        """
        if hasattr(self, 'cap'):  # Si existe el atributo `cap`:
            self.cap.release()  # Libera la cámara.
        cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV.

def main():
    """
    Función principal que inicializa el sistema y ejecuta la aplicación.
    """
    system = PredictionSystem()  # Crea una instancia del sistema de predicción.
    try:
        system.run()  # Ejecuta el sistema.
    finally:
        system.cleanup()  # Limpia los recursos al finalizar.

# Punto de entrada al script.
if __name__ == "__main__":
    main()  # Ejecuta la función principal si el script se ejecuta directamente.

