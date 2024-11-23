##Hay que instalar Optuna para correr este código por primera vez ( pip install optuna) 

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import optuna
from optuna.integration import SklearnSampler


class ModelTrainer:
    def __init__(self, model_dir="models", results_dir="analysis_results"):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_params_file = Path(model_dir) / 'model_params.json'
        self.model_file = Path(model_dir) / 'pose_model.pkl'
        self.results_dir = Path(results_dir)
        self.feature_names = []

        # Create directories if they don't exist
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self, filename: str = 'pose_data.csv') -> pd.DataFrame:
        """
        Loads the dataset from a CSV file.

        Args:
            filename (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        try:
            logging.info(f"Loading data from {filename}...")
            df = pd.read_csv(filename)
            logging.info(f"Data successfully loaded with shape {df.shape}.")
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the file: {e}")
            raise

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepares features and labels from the dataset.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            tuple: Features, labels, feature columns.
        """
        self.feature_names = [col for col in df.columns if col not in ['timestamp', 'activity']]
        X = df[self.feature_names].values
        y = self.label_encoder.fit_transform(df['activity'].values)
        return X, y

    def optimize_parameters_optuna(self, X_train_scaled, y_train, max_trials=50):
        """
        Optimizes hyperparameters using Bayesian Optimization with Optuna.

        Args:
            X_train_scaled (np.ndarray): Scaled training features.
            y_train (np.ndarray): Training labels.
            max_trials (int): Number of trials for optimization.

        Returns:
            dict: Best parameters and scores for RandomForest and XGBoost.
        """
        def rf_objective(trial):
            """Objective function for Random Forest optimization."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
            rf = RandomForestClassifier(**params)
            score = cross_val_score(rf, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='accuracy').mean()
            return score

        def xgb_objective(trial):
            """Objective function for XGBoost optimization."""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            xgb_model = xgb.XGBClassifier(**params)
            score = cross_val_score(xgb_model, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='accuracy').mean()
            return score

        # Run optimizations for both models
        rf_study = optuna.create_study(direction='maximize', sampler=SklearnSampler())
        rf_study.optimize(rf_objective, n_trials=max_trials)

        xgb_study = optuna.create_study(direction='maximize', sampler=SklearnSampler())
        xgb_study.optimize(xgb_objective, n_trials=max_trials)

        # Compare results and save the best parameters
        rf_best_params = rf_study.best_params
        rf_best_score = rf_study.best_value

        xgb_best_params = xgb_study.best_params
        xgb_best_score = xgb_study.best_value

        best_model = 'random_forest' if rf_best_score >= xgb_best_score else 'xgboost'

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

        # Save parameters to a JSON file
        with open(self.model_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)

        logging.info(f"Optimization complete. Best model: {best_model}")
        return best_params

    def train_model(self, force_optimization: bool = True):
        """
        Trains the best model based on optimized parameters.

        Args:
            force_optimization (bool): Whether to re-optimize parameters.
        """
        df = self.load_data()
        X, y = self.prepare_data(df)

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Load or optimize parameters
        if force_optimization or not self.model_params_file.exists():
            best_params = self.optimize_parameters_optuna(X_train_scaled, y_train)
        else:
            with open(self.model_params_file, 'r') as f:
                best_params = json.load(f)

        # Train the best model
        if best_params['best_model'] == 'random_forest':
            self.model = RandomForestClassifier(**best_params['random_forest']['params'])
        else:
            self.model = xgb.XGBClassifier(**best_params['xgboost']['params'], use_label_encoder=False, eval_metric='logloss')

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"Final Model Accuracy: {accuracy:.4f}")
        logging.info("\n" + classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Save the trained model
        self.save_model()

        # Analyze results
        self.analyze_results(y_test, y_pred)

    def save_model(self):
        """Saves the trained model to disk."""
        with open(self.model_file, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.label_encoder), f)
        logging.info(f"Model saved to {self.model_file}.")

    def analyze_results(self, y_test, y_pred):
        """
        Analyzes and visualizes the model's results.

        Args:
            y_test (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()

        # Feature importance for Random Forest
        if isinstance(self.model, RandomForestClassifier):
            feature_importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance.png')
            plt.close()

            logging.info(f"Feature importances saved to {self.results_dir}.")


def main():
    trainer = ModelTrainer()
    trainer.train_model(force_optimization=True)


if __name__ == "__main__":
    main()
