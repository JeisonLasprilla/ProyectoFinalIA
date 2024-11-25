import pandas as pd  # Para la manipulación y análisis de datos en estructuras como DataFrames.
import numpy as np  # Para operaciones numéricas y manejo de arrays.
import logging  # Para registrar eventos importantes durante la ejecución del programa.
from sklearn.model_selection import train_test_split, cross_val_score  # Para dividir datos y validar modelos.
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Herramientas para normalizar datos y codificar etiquetas.
from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificación basado en bosques aleatorios.
import xgboost as xgb  # Algoritmo de boosting avanzado para tareas de clasificación.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas de evaluación del modelo.
import matplotlib.pyplot as plt  # Para la creación de gráficos y visualizaciones.
import seaborn as sns  # Extensión de matplotlib para gráficos estadísticos avanzados.
from pathlib import Path  # Para gestionar rutas de archivos de manera eficiente.
import pickle  # Para serializar objetos como modelos entrenados.
import json  # Para manejar datos estructurados en formato JSON.
import optuna  # Para optimización avanzada de hiperparámetros.
from optuna.integration import SklearnSampler  # Integra Optuna con modelos de Scikit-Learn.

# Clase principal para manejar el flujo completo del modelo: desde la carga de datos hasta la evaluación.
class ModelTrainer:
    def __init__(self, model_dir="models", results_dir="analysis_results"):
        """
        Inicializa la clase configurando rutas para guardar modelos, resultados y configuraciones.
        También establece objetos clave como el escalador y el codificador de etiquetas.
        """
        self.model = None  # Modelo que será entrenado.
        self.scaler = StandardScaler()  # Escalador para normalizar características numéricas.
        self.label_encoder = LabelEncoder()  # Codificador para convertir etiquetas categóricas en valores numéricos.
        self.model_params_file = Path(model_dir) / 'model_params.json'  # Ruta para guardar parámetros optimizados.
        self.model_file = Path(model_dir) / 'pose_model.pkl'  # Ruta para guardar el modelo entrenado.
        self.results_dir = Path(results_dir)  # Carpeta para guardar resultados como métricas y gráficos.
        self.feature_names = []  # Lista para almacenar los nombres de las características.

        # Crear los directorios necesarios si no existen.
        Path(model_dir).mkdir(parents=True, exist_ok=True)  # Crea el directorio para modelos.
        self.results_dir.mkdir(parents=True, exist_ok=True)  # Crea el directorio para resultados.

        # Configuración del sistema de logging para registrar eventos importantes.
        logging.basicConfig(
            level=logging.INFO,  # Nivel de detalle del registro.
            format='%(asctime)s - %(levelname)s - %(message)s'  # Formato de los mensajes de registro.
        )
    
    def load_data(self, filename: str = 'pose_data_anga.csv') -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV.
        Args:
            filename (str): Nombre del archivo CSV que contiene los datos.
        Returns:
            pd.DataFrame: Los datos cargados en un DataFrame.
        """
        try:
            logging.info(f"Loading data from {filename}...")  # Mensaje informando la carga de datos.
            df = pd.read_csv(filename)  # Carga los datos en un DataFrame.
            logging.info(f"Data successfully loaded with shape {df.shape}.")  # Mensaje con el tamaño del dataset.
            return df
        except FileNotFoundError:  # Si el archivo no existe, registra un error.
            logging.error(f"File not found: {filename}")
            raise
        except Exception as e:  # Para cualquier otro error, registra el problema.
            logging.error(f"An error occurred while loading the file: {e}")
            raise

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepara las características y etiquetas a partir del dataset.
        Args:
            df (pd.DataFrame): Dataset cargado.
        Returns:
            tuple: Matriz de características (X) y etiquetas codificadas (y).
        """
        # Excluir columnas no relevantes y guardar los nombres de las características.
        self.feature_names = [col for col in df.columns if col not in ['timestamp', 'activity']]
        X = df[self.feature_names].values  # Extrae las características como un array NumPy.
        y = self.label_encoder.fit_transform(df['activity'].values)  # Codifica las etiquetas.
        return X, y

    def optimize_parameters_optuna(self, X_train_scaled, y_train, max_trials=50):
        """
        Optimiza los hiperparámetros de Random Forest y XGBoost utilizando Optuna.
        Args:
            X_train_scaled (array): Características normalizadas del conjunto de entrenamiento.
            y_train (array): Etiquetas del conjunto de entrenamiento.
            max_trials (int): Número máximo de intentos de optimización.
        Returns:
            dict: Los mejores parámetros para ambos modelos y el modelo recomendado.
        """
        # Función de optimización para Random Forest.
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Cantidad de árboles.
                'max_depth': trial.suggest_int('max_depth', 10, 50, log=True),  # Profundidad máxima.
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # Divisiones mínimas.
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  # Hojas mínimas.
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Características por división.
                'random_state': 42  # Fijar semilla para reproducibilidad.
            }
            rf = RandomForestClassifier(**params)  # Inicializa el modelo con los parámetros sugeridos.
            score = cross_val_score(rf, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='accuracy').mean()  # Calcula la precisión promedio.
            return score

        # Función de optimización para XGBoost.
        def xgb_objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),  # Profundidad máxima.
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Tasa de aprendizaje.
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Número de árboles.
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Peso mínimo de los nodos hijos.
                'gamma': trial.suggest_float('gamma', 0, 0.5),  # Regularización.
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Porcentaje de muestras.
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Fracción de columnas por árbol.
                'random_state': 42,  # Fijar semilla.
                'eval_metric': 'logloss',  # Métrica de evaluación.
                'use_label_encoder': False  # Desactiva el codificador por defecto.
            }
            xgb_model = xgb.XGBClassifier(**params)  # Inicializa el modelo con los parámetros sugeridos.
            score = cross_val_score(xgb_model, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='accuracy').mean()  # Precisión promedio.
            return score

        # Crea estudios para cada modelo.
        rf_study = optuna.create_study(direction='maximize', sampler=SklearnSampler())
        rf_study.optimize(rf_objective, n_trials=max_trials)  # Optimiza Random Forest.

        xgb_study = optuna.create_study(direction='maximize', sampler=SklearnSampler())
        xgb_study.optimize(xgb_objective, n_trials=max_trials)  # Optimiza XGBoost.

        # Extrae los mejores parámetros y puntajes.
        rf_best_params = rf_study.best_params
        rf_best_score = rf_study.best_value

        xgb_best_params = xgb_study.best_params
        xgb_best_score = xgb_study.best_value

        # Selecciona el mejor modelo basado en el puntaje.
        best_model = 'random_forest' if rf_best_score >= xgb_best_score else 'xgboost'

        # Guarda los parámetros optimizados en un diccionario.
        best_params = {
            'random_forest': {
                'params': rf_best_params,
                'score': rf_best_score
            },
            'xgboost': {
                'params': xgb_best_params,
                'score': xgb_best_score
            },
            'best_model': best_model
        }

        # Escribe los parámetros optimizados en un archivo JSON.
        with open(self.model_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)

        logging.info(f"Optimization complete. Best model: {best_model}")
        return best_params  # Devuelve los mejores parámetros.
    def save_evaluation_metrics(self, y_test, y_pred, class_names):
        """
        Guarda las métricas de evaluación en un archivo CSV.
        Args:
            y_test (array): Etiquetas reales del conjunto de prueba.
            y_pred (array): Etiquetas predichas por el modelo.
            class_names (list): Nombres originales de las clases.
        """
        metrics_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        metrics_df = pd.DataFrame(metrics_dict).transpose()  # Convierte el reporte a un DataFrame.
        metrics_df.to_csv(self.results_dir / 'classification_metrics.csv')  # Guarda el reporte como CSV.
        logging.info(f"Evaluation metrics saved to {self.results_dir / 'classification_metrics.csv'}")  # Registro de éxito.

    def plot_metrics_by_class(self, y_test, y_pred, class_names):
        """
        Genera un gráfico de barras con precisión, recall y F1-score por clase.
        Args:
            y_test (array): Etiquetas reales del conjunto de prueba.
            y_pred (array): Etiquetas predichas por el modelo.
            class_names (list): Nombres originales de las clases.
        """
        metrics_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Inicializa un diccionario para las métricas por clase.
        metrics_data = {
            'Precision': [],
            'Recall': [],
            'F1-score': []
        }
        classes = []  # Lista de clases presentes en el reporte.

        # Extrae métricas para cada clase.
        for class_name in class_names:
            if class_name in metrics_dict:  # Verifica que la clase exista en el reporte.
                metrics_data['Precision'].append(metrics_dict[class_name]['precision'])
                metrics_data['Recall'].append(metrics_dict[class_name]['recall'])
                metrics_data['F1-score'].append(metrics_dict[class_name]['f1-score'])
                classes.append(class_name)

        # Crea un DataFrame con las métricas por clase.
        metrics_df = pd.DataFrame(metrics_data, index=classes)

        # Genera el gráfico de barras.
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', width=0.8)  # Configura el gráfico de barras.
        plt.title('Métricas por Clase')
        plt.xlabel('Clase')
        plt.ylabel('Valor')
        plt.legend(loc='lower right')  # Coloca la leyenda en la parte inferior derecha.
        plt.xticks(rotation=45)  # Rotación de las etiquetas en el eje X.
        plt.tight_layout()  # Ajusta el diseño para evitar superposiciones.
        plt.savefig(self.results_dir / 'metrics_by_class.png')  # Guarda el gráfico.
        plt.close()  # Cierra el gráfico para liberar memoria.

    def plot_confusion_matrices(self, y_test, y_pred, class_names):
        """
        Genera matrices de confusión normalizadas y no normalizadas.
        Args:
            y_test (array): Etiquetas reales del conjunto de prueba.
            y_pred (array): Etiquetas predichas por el modelo.
            class_names (list): Nombres originales de las clases.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Crea un layout con dos gráficos.

        # Matriz sin normalizar.
        cm = confusion_matrix(y_test, y_pred)  # Calcula la matriz de confusión.
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                    yticklabels=class_names, ax=ax1)  # Visualiza la matriz en el primer gráfico.
        ax1.set_title('Matriz de Confusión (valores absolutos)')
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Real')

        # Matriz normalizada.
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')  # Calcula la matriz normalizada.
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names, ax=ax2)  # Visualiza la matriz normalizada en el segundo gráfico.
        ax2.set_title('Matriz de Confusión (normalizada)')
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Real')

        plt.tight_layout()  # Ajusta el diseño para evitar superposiciones.
        plt.savefig(self.results_dir / 'confusion_matrices.png')  # Guarda los gráficos.
        plt.close()  # Cierra la figura para liberar memoria.

    def train_model(self, force_optimization: bool = True):
        """
        Entrena el modelo, optimiza hiperparámetros y genera resultados.
        Args:
            force_optimization (bool): Si True, fuerza la optimización de hiperparámetros incluso si ya existen.
        """
        df = self.load_data()  # Carga el dataset desde el archivo especificado.
        X, y = self.prepare_data(df)  # Prepara las características y etiquetas.

        # Divide los datos en conjuntos de entrenamiento y prueba (80%-20%).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)  # Ajusta y transforma los datos de entrenamiento.
        X_test_scaled = self.scaler.transform(X_test)  # Transforma los datos de prueba.

        # Optimiza o carga los mejores hiperparámetros según el estado de `force_optimization`.
        if force_optimization or not self.model_params_file.exists():
            best_params = self.optimize_parameters_optuna(X_train_scaled, y_train)
        else:
            with open(self.model_params_file, 'r') as f:
                best_params = json.load(f)

        # Selecciona y configura el modelo basado en los mejores parámetros encontrados.
        if best_params['best_model'] == 'random_forest':
            self.model = RandomForestClassifier(**best_params['random_forest']['params'])
        else:
            self.model = xgb.XGBClassifier(**best_params['xgboost']['params'], 
                                         use_label_encoder=False, eval_metric='logloss')

        # Entrena el modelo con los datos de entrenamiento escalados.
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)  # Realiza predicciones en el conjunto de prueba.

        # Obtiene los nombres de las clases a partir del codificador.
        class_names = self.label_encoder.classes_

        # Guarda métricas y genera visualizaciones.
        self.save_evaluation_metrics(y_test, y_pred, class_names)
        self.plot_confusion_matrices(y_test, y_pred, class_names)
        self.plot_metrics_by_class(y_test, y_pred, class_names)

        # Imprime las métricas generales en consola.
        accuracy = accuracy_score(y_test, y_pred)  # Calcula la precisión.
        logging.info(f"\nAccuracy: {accuracy:.4f}")
        logging.info("\nClasification Report:")
        logging.info("\n" + classification_report(y_test, y_pred, target_names=class_names))

        # Genera importancia de características para Random Forest.
        if isinstance(self.model, RandomForestClassifier):
            self.plot_feature_importance()

        self.save_model()  # Guarda el modelo entrenado.

    def plot_feature_importance(self):
        """
        Genera un gráfico de importancia de características para Random Forest.
        """
        feature_importances = pd.DataFrame({
            'Feature': self.feature_names,  # Nombres de las características.
            'Importance': self.model.feature_importances_  # Importancia de cada característica.
        }).sort_values(by='Importance', ascending=False)  # Ordena por importancia.

        # Genera el gráfico de barras para las 20 características principales.
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()  # Ajusta el diseño.
        plt.savefig(self.results_dir / 'feature_importance.png')  # Guarda el gráfico.
        plt.close()  # Cierra la figura.

    def save_model(self):
        """
        Guarda el modelo entrenado, el escalador y el codificador de etiquetas en un archivo binario.
        """
        with open(self.model_file, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.label_encoder), f)  # Serializa y guarda.
        logging.info(f"Model saved to {self.model_file}.")  # Registra el éxito.

# Punto de entrada para ejecutar el script.
def main():
    """
    Instancia la clase `ModelTrainer` y ejecuta el flujo completo de entrenamiento y análisis.
    """
    trainer = ModelTrainer()  # Crea una instancia de `ModelTrainer`.
    trainer.train_model(force_optimization=True)  # Entrena el modelo con optimización forzada.

if __name__ == "__main__":
    main()  # Ejecuta la función principal si el script se ejecuta directamente.
