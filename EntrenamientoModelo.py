import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_params_file = 'model_params.json'
        self.model_file = 'pose_model.pkl'
        self.feature_names = None
        
    def load_data(self, filename='pose_data_anga.csv'):
        """Carga datos del archivo CSV."""
        try:
            df = pd.read_csv(filename)
            return df
        except FileNotFoundError:
            print(f"No se encontró el archivo {filename}")
            return None
            
    def optimize_parameters(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Optimiza los hiperparámetros y los guarda."""
        print("Iniciando optimización de parámetros...")
        
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        xgb_param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Optimizar Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, n_jobs=-1, verbose=2)
        rf_grid.fit(X_train_scaled, y_train)
        
        # Optimizar XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, n_jobs=-1, verbose=2)
        xgb_grid.fit(X_train_scaled, y_train)
        
        # Evaluar modelos
        rf_score = rf_grid.score(X_test_scaled, y_test)
        xgb_score = xgb_grid.score(X_test_scaled, y_test)
        
        # Guardar mejores parámetros
        best_params = {
            'random_forest': {
                'params': rf_grid.best_params_,
                'score': rf_score
            },
            'xgboost': {
                'params': xgb_grid.best_params_,
                'score': xgb_score
            },
            'best_model': 'random_forest' if rf_score >= xgb_score else 'xgboost'
        }
        
        with open(self.model_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        return best_params
        
    def train_model(self, force_optimization=True):
        """Entrena el modelo con los datos proporcionados."""
        df = self.load_data()
        if df is None:
            return
            
        # Preparar características
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'activity']]
        self.feature_names = feature_columns
        X = df[feature_columns].values
        y = self.label_encoder.fit_transform(df['activity'].values)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimizar o cargar parámetros
        if force_optimization:
            best_params = self.optimize_parameters(X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            try:
                with open(self.model_params_file, 'r') as f:
                    best_params = json.load(f)
            except FileNotFoundError:
                best_params = self.optimize_parameters(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Crear y entrenar el mejor modelo
        if best_params['best_model'] == 'random_forest':
            self.model = RandomForestClassifier(**best_params['random_forest']['params'])
        else:
            self.model = xgb.XGBClassifier(**best_params['xgboost']['params'])
        
        # Entrenar modelo final
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test_scaled)
        
        # Convertir predicciones numéricas a etiquetas
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        return X_test_scaled, y_test, y_pred, y_test_labels, y_pred_labels
    
    def save_model(self):
        """Guarda el modelo entrenado."""
        with open(self.model_file, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.label_encoder), f)
            
    def analyze_results(self, y_test, y_pred, y_test_labels, y_pred_labels):
        """Analiza y visualiza los resultados del modelo."""
        # Crear directorio para resultados
        Path("analysis_results").mkdir(exist_ok=True)
        
        # 1. Métricas generales
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        
        # Convertir el reporte de clasificación a DataFrame para mejor visualización
        report_df = pd.DataFrame(class_report).transpose()
        
        # Guardar métricas en un archivo
        report_df.to_csv('analysis_results/classification_metrics.csv')
        
        print("\nMétricas de Evaluación:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nInforme detallado por clase:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        # 2. Matriz de confusión con valores normalizados y sin normalizar
        plt.figure(figsize=(16, 6))
        
        # Matriz sin normalizar
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión (valores absolutos)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Matriz normalizada
        plt.subplot(1, 2, 2)
        cm_norm = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Matriz de Confusión (normalizada)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        plt.tight_layout()
        plt.savefig('analysis_results/confusion_matrices.png')
        plt.close()
        
        # 3. Gráfico de barras para Precision, Recall y F1-score por clase
        metrics_df = pd.DataFrame({
            'Precision': [class_report[cls]['precision'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']],
            'Recall': [class_report[cls]['recall'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']],
            'F1-score': [class_report[cls]['f1-score'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        }, index=[cls for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']])
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', width=0.8)
        plt.title('Métricas por Clase')
        plt.xlabel('Clase')
        plt.ylabel('Valor')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_results/metrics_by_class.png')
        plt.close()
        
        # 4. Importancia de características (solo para Random Forest)
        if isinstance(self.model, RandomForestClassifier):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            )
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title('Importancia de Características')
            plt.tight_layout()
            plt.savefig('analysis_results/feature_importance.png')
            plt.close()

def main():
    trainer = ModelTrainer()
    X_test_scaled, y_test, y_pred, y_test_labels, y_pred_labels = trainer.train_model(force_optimization=True)
    trainer.analyze_results(y_test, y_pred, y_test_labels, y_pred_labels)
    trainer.save_model()

if __name__ == "__main__":
    main()