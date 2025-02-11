import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

current_file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_path)

sys.path.append(os.path.abspath(os.path.join(current_file_path, '..')))
from dataset.dataset_utils import load_and_prepare_data

def setup_directories():
    """
    Crea le cartelle necessarie per il salvataggio dei risultati del modello,
    dei grafici e degli iperparametri ottimizzati.
    """
    directories = ['modelli', 'grafici', 'iperparametri/tabelle', 'iperparametri/migliori']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def prepare_data(df):
    """
    Prepara i dati per l'addestramento, dividendo il dataset in variabili 
    indipendenti (X) e variabile dipendente (y), e separando il training set 
    dal test set.
    
    Parameters::
        df (DataFrame): Il dataset contenente i dati.
    
    Returns:
        X_train (DataFrame): Le variabili indipendenti per il training set.
        X_test (DataFrame): Le variabili indipendenti per il test set.
        y_train (Series): La variabile dipendente per il training set.
        y_test (Series): La variabile dipendente per il test set.
    """
    X = df[['Gender', 'Workout_Type', 'Session_Duration_(hours)', 'Weight_(kg)', 'Age', 'Avg_BPM']]
    y = df['Calories_Burned']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def standardize_features(X_train, X_test):
    """
    Standardizza le feature utilizzando lo StandardScaler di Scikit-learn, 
    in modo che i dati abbiano media 0 e deviazione standard 1.
    
    Parameters:
        X_train (DataFrame): Il training set delle variabili indipendenti.
        X_test (DataFrame): Il test set delle variabili indipendenti.
    
    Returns:
        X_train_scaled (ndarray): Il training set standardizzato.
        X_test_scaled (ndarray): Il test set standardizzato.
        scaler (StandardScaler): L'oggetto scaler che è stato adattato ai dati di training.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_bayesian_ridge(X_train, y_train):
    """
    Addestra un modello Bayesian Ridge utilizzando GridSearchCV per trovare 
    i migliori iperparametri.
    
    Parameters:
        X_train (ndarray): Il training set delle variabili indipendenti.
        y_train (Series): Il training set della variabile dipendente.
    
    Returns:
        best_model (BayesianRidge): Il modello addestrato con i migliori parametri.
        best_params (dict): I migliori parametri trovati tramite GridSearch.
        grid_search_results (dict): I risultati completi della ricerca a griglia.
    """
    model = BayesianRidge()
    param_grid = {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4],
        'fit_intercept': [True, False]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Valuta le performance del modello addestrato utilizzando diverse metriche 
    come MAE, RMSE e R^2 sia sul training set che sul test set.
    
    Parameters:
        model (BayesianRidge): Il modello addestrato.
        X_train (ndarray): Il training set delle variabili indipendenti.
        X_test (ndarray): Il test set delle variabili indipendenti.
        y_train (Series): Il training set della variabile dipendente.
        y_test (Series): Il test set della variabile dipendente.
    
    Returns:
        metrics (dict): Un dizionario contenente le metriche per il training e il test set.
        y_test_pred (ndarray): Le predizioni sui dati di test.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {
        'train': {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R^2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R^2': r2_score(y_test, y_test_pred)
        }
    }
    return metrics, y_test_pred

def plot_metrics(metrics):
    """
    Crea un grafico a barre delle metriche di valutazione (MAE, RMSE, R^2) per 
    il training e il test set.
    
    Parameters:
        metrics (dict): Le metriche di valutazione ottenute durante la fase di valutazione.
    """
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.plot(kind='bar', figsize=(8, 5))
    plt.title("Metriche del Modello Bayesian Ridge")
    plt.ylabel("Valore")
    plt.xticks(rotation=0)
    plt.legend()
    plt.savefig('grafici/metriche_bayesian_ridge.png')
    plt.show()

def plot_predictions(y_test, y_pred):
    """
    Crea un grafico di dispersione che mostra le predizioni rispetto ai valori reali 
    del test set, evidenziando il confronto tra le due.
    
    Parameters:
        y_test (Series): I valori reali di test.
        y_pred (ndarray): Le predizioni del modello sui dati di test.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.title("Predizioni vs Valori Reali")
    plt.xlabel("Valori Reali")
    plt.ylabel("Predizioni")
    plt.savefig('grafici/predizioni_vs_valori_reali.png')
    plt.show()

def plot_errors(y_test, y_pred):
    """
    Crea un grafico a istogramma che mostra la distribuzione degli errori 
    tra i valori reali e le predizioni.
    
    Parameters:
        y_test (Series): I valori reali di test.
        y_pred (ndarray): Le predizioni del modello sui dati di test.
    """
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='orange', edgecolor='black')
    plt.title("Distribuzione degli Errori")
    plt.xlabel("Errore")
    plt.ylabel("Frequenza")
    plt.savefig('grafici/distribuzione_errori.png')
    plt.show()

def plot_prediction_trend(y_test, y_pred):
    """
    Crea un grafico che mostra l'andamento nel tempo delle predizioni 
    rispetto ai valori reali, permettendo di osservare le tendenze.
    
    Parameters:
        y_test (Series): I valori reali di test.
        y_pred (ndarray): Le predizioni del modello sui dati di test.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Valori Reali', linestyle='dashed')
    plt.plot(y_pred, label='Predizioni', alpha=0.7)
    plt.title("Andamento delle Predizioni vs Valori Reali")
    plt.xlabel("Indice")
    plt.ylabel("Calories Burned")
    plt.legend()
    plt.savefig('grafici/confronto_predizioni_valori_reali.png')
    plt.show()

def save_results(model, scaler, best_params, grid_search_results):
    """
    Salva il modello addestrato, lo scaler e i migliori parametri trovati 
    durante la grid search su file.
    
    Parameters:
        model (BayesianRidge): Il modello addestrato.
        scaler (StandardScaler): Lo scaler usato per standardizzare i dati.
        best_params (dict): I migliori parametri trovati tramite GridSearch.
        grid_search_results (dict): I risultati completi della ricerca a griglia.
    """
    joblib.dump(model, 'modelli/modello_bayesian_ridge.pkl')
    joblib.dump(scaler, 'modelli/scaler_bayesian_ridge.pkl')
    with open('iperparametri/migliori/iperparametri_bayesian_ridge.json', 'w') as json_file:
        json.dump(best_params, json_file)
    pd.DataFrame(grid_search_results).to_csv('iperparametri/tabelle/iperparametri_bayesian_ridge.csv', index=False)

def main():
    """
    Funzione principale che esegue tutte le operazioni:
    1. Crea le cartelle necessarie.
    2. Carica e prepara i dati.
    3. Addestra il modello.
    4. Valuta e visualizza i risultati.
    5. Salva il modello e i risultati.
    """
    setup_directories()
    df = load_and_prepare_data("../dataset/gym_members_exercise_tracking.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    best_model, best_params, grid_search_results = train_bayesian_ridge(X_train_scaled, y_train)
    metrics, y_test_pred = evaluate_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    plot_metrics(metrics)
    plot_predictions(y_test, y_test_pred)
    plot_errors(y_test, y_test_pred)
    plot_prediction_trend(y_test, y_test_pred)
    
    save_results(best_model, scaler, best_params, grid_search_results)
    print("Modello, scaler e parametri salvati con successo!")

if __name__ == '__main__':
    main()
