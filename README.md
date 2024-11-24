# ProyectoFinalIA

## Descripción
Este proyecto tiene como objetivo desarrollar un sistema de análisis y predicción de poses utilizando técnicas de visión por computadora y aprendizaje automático. El sistema se compone de varios módulos que permiten la recolección de datos, el entrenamiento de modelos y la predicción de actividades basadas en poses humanas.

## Estructura del Proyecto
El proyecto está compuesto por los siguientes archivos principales:

- **RecoleccionDatos.py**: Script para la recolección de datos de poses utilizando una cámara web.
- **EntrenamientoModelo.py**: Script para el entrenamiento de modelos de aprendizaje automático utilizando Grid Search.
- **EntrenamientoModeloBayesianoOptuna.py**: Script para el entrenamiento de modelos de aprendizaje automático utilizando Optuna para la optimización bayesiana de hiperparámetros.
- **ModeloPrediccion.py**: Script para la predicción de actividades basadas en poses utilizando un modelo previamente entrenado.
- **pose_data.csv**: Archivo CSV que contiene los datos de poses recolectados.
- **model_params.json**: Archivo JSON que contiene los mejores hiperparámetros encontrados durante el entrenamiento.
- **pose_model.pkl**: Archivo que contiene el modelo entrenado.

## Requisitos
Antes de ejecutar los scripts, asegúrate de tener instaladas las siguientes dependencias:
```bash
pip install -r requirements.txt
```


## Guía de Uso
## Recolección de Datos
Para recolectar datos de poses, ejecuta el script RecoleccionDatos.py. Este script abrirá una interfaz gráfica que te permitirá grabar diferentes actividades utilizando una cámara web.

- Selecciona la actividad que deseas grabar en el selector de actividades.
- Haz clic en Iniciar Grabación para comenzar a grabar.
- Realiza la actividad frente a la cámara.
- Haz clic en Detener Grabación para finalizar la grabación y guardar los datos en pose_data.csv.


## Entrenamiento del Modelo

**Usando Grid Search**

- Para entrenar el modelo utilizando Grid Search para la optimización de hiperparámetros:

```bash
python EntrenamientoModelo.py 
```

- El script cargará los datos de pose_data.csv, optimizará los hiperparámetros y entrenará el mejor modelo.
- Los resultados del entrenamiento se guardarán en model_params.json y pose_model.pkl.

**Usando Optuna**

Para entrenar el modelo utilizando Optuna para la optimización bayesiana de hiperparámetros:

```bash
python EntrenamientoModeloBayesianoOptuna.py
```

- El script cargará los datos de pose_data.csv, optimizará los hiperparámetros utilizando Optuna y entrenará el mejor modelo.
- Los resultados del entrenamiento se guardarán en model_params.json y pose_model.pkl.

## Predicción de Actividades

Para predecir actividades basadas en poses utilizando el modelo entrenado:

```bash
python ModeloPrediccion.py
```

- Este script abrirá una interfaz gráfica que utilizará la cámara web para capturar poses en tiempo real y predecir la actividad correspondiente utilizando el modelo entrenado.

