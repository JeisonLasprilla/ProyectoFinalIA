# pip install mediapipe opencv-python numpy pandas matplotlib scikit-learn seaborn pillow xgboost

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class PoseAnalysisSystem:
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Actividades a detectar
        self.activities = [
            "walking_towards",
            "walking_away",
            "turning",
            "sitting",
            "standing"
        ]
        
        # Estructura de datos
        self.landmarks_data = []
        self.current_activity = None
        self.recording = False
        
        # Modelo y escalador
        self.model = None
        self.scaler = StandardScaler()
        
        # Rutas de archivos
        self.model_params_file = 'model_params.json'
        self.model_file = 'pose_model.pkl'
        
        # Buffer para análisis en tiempo real
        self.frame_buffer = []
        self.buffer_size = 30
            
            
    def create_gui(self):
        """Crea la interfaz gráfica para la captura de datos y visualización."""
        self.root = tk.Tk()
        self.root.title("Sistema de Análisis de Poses")
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2)
        
        # Controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Selector de actividad
        self.activity_var = tk.StringVar()
        activity_selector = ttk.Combobox(
            controls_frame, 
            textvariable=self.activity_var,
            values=self.activities
        )
        activity_selector.grid(row=0, column=0, padx=5)
        
        # Botones
        self.record_btn = ttk.Button(
            controls_frame,
            text="Iniciar Grabación",
            command=self.toggle_recording
        )
        self.record_btn.grid(row=0, column=1, padx=5)
        
        # Estado actual
        self.status_label = ttk.Label(main_frame, text="Estado: Esperando")
        self.status_label.grid(row=2, column=0, columnspan=2)
        
        # Variables de control
        self.is_predicting = False
        self.cap = cv2.VideoCapture(0)
        
        # Iniciar actualización de video
        self.update_video()
        
    def update_video(self):
        """Actualiza el frame de video en la GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame, landmarks = self.process_frame(frame)
            
            if self.recording and landmarks is not None:
                self.collect_data(landmarks)
            
            if self.is_predicting and landmarks is not None:
                self.predict_activity(landmarks)
            
            # Convertir frame para GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(10, self.update_video)
        
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
            
            # Calcular y mostrar ángulos importantes
            landmarks_dict = self._extract_landmarks(results.pose_landmarks)
            angles = self._calculate_key_angles(landmarks_dict)
            
            # Mostrar ángulos en el frame
            y_pos = 30
            for angle_name, angle_value in angles.items():
                cv2.putText(
                    frame,
                    f"{angle_name}: {angle_value:.1f}°",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_pos += 20
        
        return frame, results.pose_landmarks if results.pose_landmarks else None
        
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
        
    def collect_data(self, landmarks):
        """Recolecta datos de landmarks con etiquetas."""
        if self.current_activity:
            landmarks_dict = self._extract_landmarks(landmarks)
            angles = self._calculate_key_angles(landmarks_dict)
            
            data_point = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'activity': self.current_activity,
                **landmarks_dict,
                **angles
            }
            
            self.landmarks_data.append(data_point)
            
    def toggle_recording(self):
        """Alterna el estado de grabación."""
        self.recording = not self.recording
        self.current_activity = self.activity_var.get() if self.recording else None
        
        if self.recording:
            self.record_btn.configure(text="Detener Grabación")
            self.status_label.configure(
                text=f"Grabando actividad: {self.current_activity}"
            )
        else:
            self.record_btn.configure(text="Iniciar Grabación")
            self.status_label.configure(text="Grabación detenida")
            self.save_data()
            
    def save_data(self, filename='pose_data.csv'):
        """Guarda los datos recolectados."""
        if self.landmarks_data:
            df = pd.DataFrame(self.landmarks_data)
            df.to_csv(filename, index=False)
            print(f"Datos guardados en {filename}")
            
    def analyze_data(self):
        """Realiza análisis exploratorio de datos."""
        if not self.landmarks_data:
            print("No hay datos para analizar")
            return
            
        df = pd.DataFrame(self.landmarks_data)
        
        # Crear directorio para gráficos
        Path("analysis_results").mkdir(exist_ok=True)
        
        # 1. Distribución de actividades
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='activity')
        plt.title('Distribución de Actividades')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_results/activity_distribution.png')
        plt.close()
        
        # 2. Análisis de ángulos por actividad
        angle_columns = ['Left Knee', 'Right Knee', 'Trunk Inclination']
        for angle in angle_columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='activity', y=angle)
            plt.title(f'Distribución de {angle} por Actividad')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'analysis_results/{angle.lower().replace(" ", "_")}_distribution.png')
            plt.close()
        
        # 3. Trayectorias de movimiento
        activities = df['activity'].unique()
        for activity in activities:
            activity_data = df[df['activity'] == activity]
            
            plt.figure(figsize=(12, 6))
            plt.plot(activity_data['NOSE_x'], activity_data['NOSE_y'])
            plt.title(f'Trayectoria de Movimiento - {activity}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.gca().invert_yaxis()  # Invertir eje Y para coincidir con coordenadas de imagen
            plt.tight_layout()
            plt.savefig(f'analysis_results/trajectory_{activity.lower()}.png')
            plt.close()
            
    def prepare_features(self, landmarks_dict, angles):
        """Prepara características para el modelo."""
        features = []
        for key in landmarks_dict.keys():
            features.append(landmarks_dict[key])
        for key in angles.keys():
            features.append(angles[key])
        return np.array(features).reshape(1, -1)
            
    
    def run(self):
        """Ejecuta la aplicación."""
        self.create_gui()
        self.root.mainloop()
        
    def cleanup(self):
        """Limpia recursos."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
def main():
    system = PoseAnalysisSystem()
    try:
        system.run()
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()