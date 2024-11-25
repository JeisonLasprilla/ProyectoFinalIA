import pandas as pd  # Librería para manipulación y análisis de datos estructurados.
import numpy as np  # Librería para operaciones matemáticas y arreglos multidimensionales.
from sklearn.model_selection import train_test_split, GridSearchCV  # Utilidad para dividir datos y optimizar hiperparámetros.
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Herramientas para normalización y codificación de etiquetas.
from sklearn.ensemble import RandomForestClassifier  # Modelo basado en ensamble de árboles.
import xgboost as xgb  # Modelo avanzado de boosting.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas para evaluar el rendimiento.
import matplotlib.pyplot as plt  # Librería para generación de gráficos.
import seaborn as sns  # Extensión de matplotlib para visualizaciones más avanzadas.
from pathlib import Path  # Manejo de rutas de archivos.
import pickle  # Serialización de objetos para guardar modelos entrenados.
import json  # Manejo de estructuras de datos en formato JSON.

# Definición de la clase para manejar el flujo de entrenamiento, optimización y evaluación del modelo.
class ModelTrainer:
    def __init__(self):
        """Inicializa la clase con atributos básicos como el modelo, escalador, codificador y rutas de archivos."""
        self.model = None  # Lugar para almacenar el modelo entrenado.
        self.scaler = StandardScaler()  # Normalizador de datos numéricos para escalar entre valores similares.
        self.label_encoder = LabelEncoder()  # Codificador para transformar etiquetas categóricas en valores numéricos.
        self.model_params_file = 'model_params.json'  # Archivo para guardar los mejores hiperparámetros.
        self.model_file = 'pose_model.pkl'  # Archivo donde se guardará el modelo entrenado.
        self.feature_names = None  # Lista de nombres de las características (columnas del dataset).

    def load_data(self, filename='pose_data_anga.csv'):
        """
        Carga el dataset desde un archivo CSV.
        Args:
            filename (str): Nombre del archivo CSV a cargar.
        Returns:
            pd.DataFrame o None: Devuelve un DataFrame con los datos si el archivo existe; de lo contrario, None.
        """
        try:
            df = pd.read_csv(filename)  # Carga el archivo CSV en un DataFrame.
            return df  # Devuelve el DataFrame cargado.
        except FileNotFoundError:  # Manejo de error si el archivo no existe.
            print(f"No se encontró el archivo {filename}")
            return None  # Devuelve None si hay un error.

    def optimize_parameters(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """
        Encuentra los mejores hiperparámetros para los modelos Random Forest y XGBoost usando GridSearchCV.
        Args:
            X_train_scaled: Datos de entrenamiento normalizados.
            X_test_scaled: Datos de prueba normalizados.
            y_train: Etiquetas de entrenamiento.
            y_test: Etiquetas de prueba.
        Returns:
            dict: Diccionario con los mejores parámetros y el modelo recomendado.
        """
        print("Iniciando optimización de parámetros...")  # Mensaje para indicar el inicio del proceso.

        # Definición del grid de hiperparámetros para Random Forest.
        rf_param_grid = {
            'n_estimators': [100, 200, 300],  # Cantidad de árboles en el bosque.
            'max_depth': [10, 20, 30, None],  # Profundidad máxima de los árboles.
            'min_samples_split': [2, 5, 10],  # Mínimo de muestras necesarias para dividir un nodo.
            'min_samples_leaf': [1, 2, 4],  # Mínimo de muestras requeridas para ser una hoja.
            'max_features': ['sqrt', 'log2']  # Cantidad de características consideradas en cada división.
        }

        # Definición del grid de hiperparámetros para XGBoost.
        xgb_param_grid = {
            'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol.
            'learning_rate': [0.01, 0.1, 0.3],  # Tasa de aprendizaje.
            'n_estimators': [100, 200, 300],  # Número de árboles en el ensamble.
            'min_child_weight': [1, 3, 5],  # Peso mínimo de los nodos hijos.
            'gamma': [0, 0.1, 0.2],  # Regularización para evitar sobreajuste.
            'subsample': [0.8, 0.9, 1.0]  # Porcentaje de muestras utilizadas en cada iteración.
        }

        # Optimización de Random Forest con GridSearchCV.
        rf = RandomForestClassifier(random_state=42)  # Modelo base de Random Forest.
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, n_jobs=-1, verbose=2)  # Configuración de la búsqueda.
        rf_grid.fit(X_train_scaled, y_train)  # Ajuste de los datos de entrenamiento.

        # Optimización de XGBoost con GridSearchCV.
        xgb_model = xgb.XGBClassifier(random_state=42)  # Modelo base de XGBoost.
        xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, n_jobs=-1, verbose=2)  # Configuración de la búsqueda.
        xgb_grid.fit(X_train_scaled, y_train)  # Ajuste de los datos de entrenamiento.

        # Evaluación del desempeño de ambos modelos en el conjunto de prueba.
        rf_score = rf_grid.score(X_test_scaled, y_test)  # Precisión de Random Forest.
        xgb_score = xgb_grid.score(X_test_scaled, y_test)  # Precisión de XGBoost.

        # Guardar los mejores parámetros y el modelo con mejor desempeño.
        best_params = {
            'random_forest': {
                'params': rf_grid.best_params_,  # Mejores parámetros de Random Forest.
                'score': rf_score  # Puntaje del modelo.
            },
            'xgboost': {
                'params': xgb_grid.best_params_,  # Mejores parámetros de XGBoost.
                'score': xgb_score  # Puntaje del modelo.
            },
            'best_model': 'random_forest' if rf_score >= xgb_score else 'xgboost'  # Selección del mejor modelo.
        }

        # Guardar los parámetros optimizados en un archivo JSON.
        with open(self.model_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)

        return best_params  # Devolver los mejores parámetros.

    def train_model(self, force_optimization=True):
        """
        Entrena el modelo con los datos proporcionados.
        Args:
            force_optimization (bool): Si es True, se realiza una optimización de hiperparámetros.
        Returns:
            tuple: Conjunto de prueba, etiquetas reales, predicciones, etiquetas reales (descodificadas), predicciones (descodificadas).
        """
        df = self.load_data()  # Carga los datos del archivo especificado.
        if df is None:  # Verifica si no se pudo cargar el archivo.
            return  # Termina el entrenamiento si no hay datos.

        # Definir columnas de características (excluyendo columnas irrelevantes como 'timestamp' y 'activity').
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'activity']]
        self.feature_names = feature_columns  # Guarda los nombres de las características.
        X = df[feature_columns].values  # Extrae las características como un arreglo NumPy.
        y = self.label_encoder.fit_transform(df['activity'].values)  # Codifica las etiquetas de actividades en valores numéricos.

        # División del dataset en conjuntos de entrenamiento y prueba (80%-20%).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Escala los datos para normalizarlos entre valores comparables.
        X_train_scaled = self.scaler.fit_transform(X_train)  # Ajusta y transforma los datos de entrenamiento.
        X_test_scaled = self.scaler.transform(X_test)  # Transforma los datos de prueba usando el mismo escalador.

        # Optimización de hiperparámetros, si está habilitada.
        if force_optimization:
            best_params = self.optimize_parameters(X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            try:
                # Intenta cargar los parámetros previamente guardados.
                with open(self.model_params_file, 'r') as f:
                    best_params = json.load(f)
            except FileNotFoundError:
                # Si no se encuentran parámetros guardados, realiza la optimización.
                best_params = self.optimize_parameters(X_train_scaled, X_test_scaled, y_train, y_test)

        # Crea y entrena el modelo con los mejores parámetros encontrados.
        if best_params['best_model'] == 'random_forest':
            self.model = RandomForestClassifier(**best_params['random_forest']['params'])
        else:
            self.model = xgb.XGBClassifier(**best_params['xgboost']['params'])

        # Ajusta el modelo a los datos de entrenamiento escalados.
        self.model.fit(X_train_scaled, y_train)

        # Realiza predicciones en el conjunto de prueba.
        y_pred = self.model.predict(X_test_scaled)

        # Descifra las etiquetas numéricas a sus valores originales.
        y_test_labels = self.label_encoder.inverse_transform(y_test)  # Etiquetas reales.
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)  # Etiquetas predichas.

        # Devuelve los datos para su análisis posterior.
        return X_test_scaled, y_test, y_pred, y_test_labels, y_pred_labels

    def save_model(self):
        """
        Guarda el modelo entrenado, el escalador y el codificador de etiquetas en un archivo binario.
        """
        with open(self.model_file, 'wb') as f:  # Abre el archivo en modo escritura binaria.
            pickle.dump((self.model, self.scaler, self.label_encoder), f)  # Serializa y guarda los objetos.

    def analyze_results(self, y_test, y_pred, y_test_labels, y_pred_labels):
        """
        Analiza y visualiza los resultados del modelo.
        Args:
            y_test: Etiquetas reales del conjunto de prueba.
            y_pred: Etiquetas predichas por el modelo.
            y_test_labels: Etiquetas reales descodificadas.
            y_pred_labels: Etiquetas predichas descodificadas.
        """
        # Crear un directorio para almacenar los resultados, si no existe.
        Path("analysis_results").mkdir(exist_ok=True)

        # 1. Métricas generales.
        accuracy = accuracy_score(y_test, y_pred)  # Calcula la precisión general.
        class_report = classification_report(y_test_labels, y_pred_labels, output_dict=True)  # Genera un informe detallado.

        # Convertir el informe de clasificación a un DataFrame para guardarlo como CSV.
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv('analysis_results/classification_metrics.csv')  # Guarda las métricas en un archivo CSV.

        # Imprime las métricas generales en la consola.
        print("\nMétricas de Evaluación:")
        print(f"Accuracy: {accuracy:.4f}")  # Precisión global.
        print("\nInforme detallado por clase:")
        print(classification_report(y_test_labels, y_pred_labels))  # Informe detallado.

        # 2. Matriz de confusión (valores absolutos y normalizados).
        plt.figure(figsize=(16, 6))  # Tamaño de la figura.

        # Matriz sin normalizar.
        plt.subplot(1, 2, 1)  # Primera gráfica de dos.
        cm = confusion_matrix(y_test_labels, y_pred_labels)  # Calcula la matriz de confusión.
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Visualiza la matriz de confusión.
        plt.title('Matriz de Confusión (valores absolutos)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')

        # Matriz normalizada.
        plt.subplot(1, 2, 2)  # Segunda gráfica.
        cm_norm = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')  # Matriz normalizada.
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')  # Visualiza la matriz normalizada.
        plt.title('Matriz de Confusión (normalizada)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')

        # Ajuste de la figura y guardado.
        plt.tight_layout()
        plt.savefig('analysis_results/confusion_matrices.png')  # Guarda la gráfica.
        plt.close()

        # 3. Gráfico de barras para Precision, Recall y F1-score por clase.
        metrics_df = pd.DataFrame({
            'Precision': [class_report[cls]['precision'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']],
            'Recall': [class_report[cls]['recall'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']],
            'F1-score': [class_report[cls]['f1-score'] for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        }, index=[cls for cls in class_report if cls not in ['accuracy', 'macro avg', 'weighted avg']])

        # Gráfica de métricas.
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', width=0.8)  # Crea un gráfico de barras.
        plt.title('Métricas por Clase')
        plt.xlabel('Clase')
        plt.ylabel('Valor')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_results/metrics_by_class.png')  # Guarda la gráfica.
        plt.close()

        # 4. Importancia de características (solo para Random Forest).
        if isinstance(self.model, RandomForestClassifier):  # Verifica el tipo de modelo.
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,  # Nombres de las características.
                'importance': self.model.feature_importances_  # Importancia calculada.
            })
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False  # Ordena por importancia.
            )

            # Gráfica de importancia de características.
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))  # Muestra las 20 más importantes.
            plt.title('Importancia de Características')
            plt.tight_layout()
            plt.savefig('analysis_results/feature_importance.png')  # Guarda la gráfica.
            plt.close()
# Definición de la función principal que ejecuta el flujo completo del entrenamiento y análisis.
def main():
    """
    Función principal que realiza el entrenamiento del modelo, analiza los resultados y guarda el modelo entrenado.
    """
    trainer = ModelTrainer()  # Instancia la clase `ModelTrainer`.

    # Llama a la función `train_model` para entrenar el modelo con optimización de hiperparámetros.
    X_test_scaled, y_test, y_pred, y_test_labels, y_pred_labels = trainer.train_model(force_optimization=True)

    # Analiza los resultados del modelo entrenado, incluyendo métricas, gráficos y reportes.
    trainer.analyze_results(y_test, y_pred, y_test_labels, y_pred_labels)

    # Guarda el modelo entrenado, junto con el escalador y el codificador de etiquetas.
    trainer.save_model()

# Punto de entrada al script.
if __name__ == "__main__":
    """
    Comprueba si el script se está ejecutando directamente (no importado como módulo),
    y en ese caso, llama a la función principal.
    """
    main()