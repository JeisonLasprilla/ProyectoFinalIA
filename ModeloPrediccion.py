import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class PredictionSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cargar modelo entrenado
        self.load_model()
        
        # Buffer para predicciones
        self.prediction_buffer = []
        self.buffer_size = 5
        
    def create_gui(self):
        """Crea la interfaz gráfica para la predicción."""
        self.root = tk.Tk()
        self.root.title("Sistema de Predicción de Poses")
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2)
        
        # Panel de predicción
        pred_frame = ttk.LabelFrame(main_frame, text="Predicción", padding="5")
        pred_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.prediction_label = ttk.Label(
            pred_frame,
            text="Actividad detectada: -",
            font=('Arial', 14, 'bold')
        )
        self.prediction_label.grid(row=0, column=0)
        
        # Confianza de predicción
        self.confidence_label = ttk.Label(
            pred_frame,
            text="Confianza: -",
            font=('Arial', 12)
        )
        self.confidence_label.grid(row=1, column=0)
        
        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(
            controls_frame,
            text="Iniciar/Detener",
            command=self.toggle_prediction
        ).grid(row=0, column=0, padx=5)
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        self.is_predicting = False
        
    def load_model(self):
        """Carga el modelo entrenado."""
        try:
            with open('pose_model.pkl', 'rb') as f:
                self.model, self.scaler, self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                "No se encontró el modelo entrenado (pose_model.pkl)"
            )
            self.model = None
            
    def process_frame(self, frame):
        """Procesa un frame y extrae landmarks."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            landmarks_dict = self._extract_landmarks(results.pose_landmarks)
            angles = self._calculate_key_angles(landmarks_dict)
            
            if self.is_predicting:
                self.predict_activity(landmarks_dict, angles)
            
        return frame
        
    def _extract_landmarks(self, landmarks):
        """Extrae coordenadas de landmarks importantes."""
        keypoints = {}
        
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
        
        for name, landmark in important_landmarks.items():
            keypoints[f'{name}_x'] = landmarks.landmark[landmark].x
            keypoints[f'{name}_y'] = landmarks.landmark[landmark].y
            keypoints[f'{name}_z'] = landmarks.landmark[landmark].z
            
        return keypoints
        
    def _calculate_key_angles(self, landmarks):
        """Calcula ángulos importantes del cuerpo."""
        angles = {}
        
        # Ángulo de rodilla izquierda
        left_knee_angle = self._calculate_angle(
            [landmarks['LEFT_HIP_x'], landmarks['LEFT_HIP_y']],
            [landmarks['LEFT_KNEE_x'], landmarks['LEFT_KNEE_y']],
            [landmarks['LEFT_ANKLE_x'], landmarks['LEFT_ANKLE_y']]
        )
        angles['Left Knee'] = left_knee_angle
        
        # Ángulo de rodilla derecha
        right_knee_angle = self._calculate_angle(
            [landmarks['RIGHT_HIP_x'], landmarks['RIGHT_HIP_y']],
            [landmarks['RIGHT_KNEE_x'], landmarks['RIGHT_KNEE_y']],
            [landmarks['RIGHT_ANKLE_x'], landmarks['RIGHT_ANKLE_y']]
        )
        angles['Right Knee'] = right_knee_angle
        
        # Inclinación del tronco
        trunk_angle = self._calculate_angle(
            [(landmarks['LEFT_SHOULDER_x'] + landmarks['RIGHT_SHOULDER_x'])/2,
             (landmarks['LEFT_SHOULDER_y'] + landmarks['RIGHT_SHOULDER_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
             (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
             (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2 + 0.1]
        )
        angles['Trunk Inclination'] = trunk_angle
        
        return angles
        
    def _calculate_angle(self, a, b, c):
        """Calcula el ángulo entre tres puntos."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
        
    def prepare_features(self, landmarks_dict, angles):
        """Prepara características para el modelo."""
        features = []
        for key in landmarks_dict.keys():
            features.append(landmarks_dict[key])
        for key in angles.keys():
            features.append(angles[key])
        return np.array(features).reshape(1, -1)
        
    def predict_activity(self, landmarks_dict, angles):
        """Realiza predicción de la actividad."""
        if self.model is None:
            return
            
        # Preparar características
        features = self.prepare_features(landmarks_dict, angles)
        
        # Escalar características
        features_scaled = self.scaler.transform(features)
        
        # Predicción
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Obtener probabilidades
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max() * 100
        
        # Actualizar buffer de predicciones
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # Obtener predicción más frecuente
        from collections import Counter
        final_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
        
        # Actualizar GUI
        self.prediction_label.configure(
            text=f"Actividad detectada: {final_prediction}"
        )
        self.confidence_label.configure(
            text=f"Confianza: {confidence:.1f}%"
        )
        
    def update_video(self):
        """Actualiza el frame de video en la GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)
            
            # Convertir frame para GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(10, self.update_video)
        
    def toggle_prediction(self):
        """Alterna el modo de predicción."""
        self.is_predicting = not self.is_predicting
        if not self.is_predicting:
            self.prediction_label.configure(text="Actividad detectada: -")
            self.confidence_label.configure(text="Confianza: -")
            self.prediction_buffer.clear()
            
    def run(self):
        """Ejecuta la aplicación."""
        self.create_gui()
        self.update_video()
        self.root.mainloop()
        
    def cleanup(self):
        """Limpia recursos."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    system = PredictionSystem()
    try:
        system.run()
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()