import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

current_file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_path)

sys.path.append(os.path.abspath(os.path.join(current_file_path, '..')))
from dataset.dataset_utils import *

def setup_directories():
    """
    Crea le cartelle necessarie per il salvataggio dei modelli, grafici e risultati.
    
    Crea le seguenti cartelle:
    - 'modelli': per salvare i modelli addestrati.
    - 'grafici': per salvare i grafici delle metriche.
    - 'iperparametri/tabelle': per salvare i risultati della ricerca sugli iperparametri.
    - 'iperparametri/migliori': per salvare i migliori iperparametri.
    """
    directories = ['modelli', 'grafici', 'iperparametri/tabelle', 'iperparametri/migliori']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def prepare_data(df):
    """
    Prepara i dati per l'addestramento e il test dei modelli.
    
    Parametri:
    - df (DataFrame): il dataset da preparare.
    
    Restituisce:
    - X_train (DataFrame): le feature del training set.
    - X_test (DataFrame): le feature del test set.
    - y_train (Series): il target del training set.
    - y_test (Series): il target del test set.
    """
    X = df[['Gender', 'Workout_Type', 'Session_Duration_(hours)', 'Weight_(kg)', 'Age', 'Avg_BPM']]
    y = df['Calories_Burned']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    """
    Addestra un modello di regressione lineare.
    
    Parametri:
    - X_train (DataFrame): le feature del training set.
    - y_train (Series): il target del training set.
    
    Restituisce:
    - model (LinearRegression): il modello addestrato.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_model_with_grid_search(model_class, param_grid, X_train, y_train):
    """
    Esegue una ricerca sugli iperparametri (Grid Search) per addestrare un modello.
    
    Parametri:
    - model_class (class): la classe del modello da addestrare (e.g., RandomForestRegressor).
    - param_grid (dict): la griglia di ricerca per gli iperparametri.
    - X_train (DataFrame): le feature del training set.
    - y_train (Series): il target del training set.
    
    Restituisce:
    - best_model (model): il miglior modello addestrato.
    - best_params (dict): i migliori parametri trovati dal grid search.
    - cv_results (dict): i risultati della ricerca sugli iperparametri.
    """
    grid_search = GridSearchCV(model_class(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Migliori parametri trovati per {model_class.__name__}:", best_params)
    best_model = model_class(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params, grid_search.cv_results_

def get_hyperparameters():
    """
    Definisce i parametri per la ricerca sugli iperparametri per Random Forest e Gradient Boosting.
    
    Restituisce:
    - dict: la griglia di ricerca per ogni modello.
    """
    return {
        'random_forest': {
            'n_estimators': [200, 300],
            'max_depth': [10, 15],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5, 10]
        }
    }

def evaluate_models(models, X_test, y_test, X_train, y_train):
    """
    Esegue la valutazione dei modelli usando MAE e RMSE su set di training e test.
    
    Parametri:
    - models (dict): i modelli da valutare.
    - X_test (DataFrame): le feature del test set.
    - y_test (Series): il target del test set.
    - X_train (DataFrame): le feature del training set.
    - y_train (Series): il target del training set.
    
    Restituisce:
    - metrics (dict): un dizionario contenente le metriche per ogni modello sui set di training e test.
    """
    metrics = {'test': [], 'train': []}
    for name, model in models.items():
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        metrics['test'].append({
            'Modello': name,
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
        })
        metrics['train'].append({
            'Modello': name,
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train))
        })
    return metrics

def plot_metrics(metrics, title, filename):
    """
    Crea e salva un grafico delle metriche di errore (MAE e RMSE) per ogni modello.
    
    Parametri:
    - metrics (dict): le metriche da tracciare (contenente 'MAE' e 'RMSE' per ogni modello).
    - title (str): il titolo del grafico.
    - filename (str): il percorso dove salvare il grafico.
    """
    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(df_metrics))
    plt.bar(x - bar_width/2, df_metrics['MAE'], bar_width, label='MAE', color='blue')
    plt.bar(x + bar_width/2, df_metrics['RMSE'], bar_width, label='RMSE', color='red')
    plt.title(title)
    plt.ylabel('Errore')
    plt.xticks(x, df_metrics['Modello'], rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def save_results(models, best_params, results):
    """
    Salva i modelli addestrati, i migliori iperparametri e i risultati della grid search.
    
    Parametri:
    - models (dict): i modelli addestrati.
    - best_params (dict): i migliori parametri per ogni modello.
    - results (dict): i risultati della grid search per ogni modello.
    """
    for name, model in models.items():
        joblib.dump(model, f'modelli/modello_{name}.pkl')
    
    for name, params in best_params.items():
        with open(f'iperparametri/migliori/iperparametri_{name}.json', 'w') as json_file:
            json.dump(params, json_file)
        
    for name, result in results.items():
        pd.DataFrame(result).to_csv(f'iperparametri/tabelle/iperparametri_{name}.csv', index=False)

def main():
    """
    Funzione principale che esegue il flusso completo del processo di addestramento e valutazione del modello.
    
    1. Imposta le directory necessarie.
    2. Carica e prepara il dataset.
    3. Esegue il training dei modelli.
    4. Valuta le performance dei modelli sui set di training e test.
    5. Crea i grafici delle metriche di errore.
    6. Salva i modelli, i migliori iperparametri e i risultati.
    """
    setup_directories()
    df = load_and_prepare_data("../dataset/gym_members_exercise_tracking.csv")
    # rimuovi la directory principale dal path
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    linear_model = train_linear_regression(X_train_scaled, y_train)
    hyperparams = get_hyperparameters()

    rf_model, rf_params, rf_results = train_model_with_grid_search(RandomForestRegressor, hyperparams['random_forest'], X_train_scaled, y_train)
    gb_model, gb_params, gb_results = train_model_with_grid_search(GradientBoostingRegressor, hyperparams['gradient_boosting'], X_train_scaled, y_train)

    models = {'Linear Regression': linear_model, 'Random Forest': rf_model, 'Gradient Boosting': gb_model}
    best_params = {'random_forest': rf_params, 'gradient_boosting': gb_params}
    results = {'random_forest': rf_results, 'gradient_boosting': gb_results}

    metrics = evaluate_models(models, X_test_scaled, y_test, X_train_scaled, y_train)

    plot_metrics(metrics['test'], "Performance sui dati di test", "grafici/test_set_metriche.png")
    plot_metrics(metrics['train'], "Performance sui dati di training", "grafici/training_set_metriche.png")

    save_results(models, best_params, results)
    print("Modelli addestrati e salvati con successo!")

if __name__ == '__main__':
    main()