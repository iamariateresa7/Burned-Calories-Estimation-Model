import pandas as pd
import joblib
import os

def load_models():
    """
    Carica i modelli di machine learning e lo scaler da file salvati.

    Questa funzione carica i modelli di regressione (Linear Regression, Random Forest, Gradient Boosting, Bayesian Ridge)
    e lo scaler, utilizzati per normalizzare i dati in ingresso. Inoltre, carica una mappatura dei workout.

    Returns:
        dict: un dizionario contenente i modelli caricati.
        scaler: lo scaler per la normalizzazione dei dati.
        workout_mapping: una mappatura dei tipi di allenamento.
    """
    supervised_learn_path = 'apprendimento_supervisionato/modelli/'
    probabilistic_learn_path = 'apprendimento_probabilistico/modelli/'
    models = {
        "linear": joblib.load(f'{supervised_learn_path}modello_Linear Regression.pkl'),
        "random_forest": joblib.load(f'{supervised_learn_path}modello_Random Forest.pkl'),
        "gradient_boosting": joblib.load(f'{supervised_learn_path}modello_Gradient boosting.pkl'),
        "bayesian_ridge": joblib.load(f'{probabilistic_learn_path}modello_bayesian_ridge.pkl'),
    }
    scaler = joblib.load(f'{supervised_learn_path}scaler.pkl')
    workout_mapping = joblib.load(f'{supervised_learn_path}workout_mapping.pkl')
    return models, scaler, workout_mapping

def query_prolog(weight, height, duration):
    """
    Interroga la base di conoscenza Prolog per inferire il tipo di allenamento,
    l'intensità e la durata ottimale in base ai parametri di input.

    Parameters:
        weight (float): il peso dell'utente (in kg).
        height (float): l'altezza dell'utente (in metri).
        duration (float): la durata della sessione di allenamento (in ore).

    Returns:
        tuple: una tupla contenente il tipo di allenamento, l'intensità e la durata ottimale.
    """
    os.chdir("kb") # To solve the bug in the pyswipl library
    from pyswip import Prolog

    prolog = Prolog()
    prolog.consult("kb.pl")
    
    workout_result = list(prolog.query(f"recommended_workout({weight}, {height}, Workout)"))
    workout_type = workout_result[0]['Workout'] if workout_result else None
    
    intensity_result = list(prolog.query(f"recommended_intensity({weight}, {height}, Intensity, {duration})"))
    intensity = intensity_result[0]['Intensity'] if intensity_result else None
    
    duration_result = list(prolog.query(f"optimal_duration({weight}, {height}, {duration}, OptimalDuration)"))
    optimal_duration = duration_result[0]['OptimalDuration'] if duration_result else None
    
    return workout_type, intensity, optimal_duration

def print_prolog_results(workout_mapping, workout_type, intensity, optimal_duration):
    """
    Stampa i risultati dell'inferenza fatta dalla base di conoscenza Prolog.

    Parameters:
        workout_mapping (dict): una mappatura dei tipi di allenamento.
        workout_type (int): il tipo di allenamento raccomandato.
        intensity (str): il livello di intensità raccomandato.
        optimal_duration (float): la durata ottimale dell'allenamento.
    """
    print("\n=== Risultati dell'Inferenza Prolog ===")
    print(f"Workout consigliato: {workout_mapping.get(workout_type, 'Sconosciuto')}")
    print(f"Intensità stimata: {intensity}")
    print(f"Durata ottimale: {optimal_duration} ore")

def get_user_input(workout_mapping):
    """
    Richiede all'utente di inserire i dati necessari per l'elaborazione delle predizioni.

    Parameters:
        workout_mapping (dict): una mappatura dei tipi di allenamento.

    Returns:
        tuple: una tupla contenente i dati inseriti dall'utente (genere, età, peso, altezza, battiti medi, tipo di allenamento e durata della sessione).
    """
    gender = int(input("Inserisci il genere (0 = Maschio, 1 = Femmina): "))
    
    print("\nSeleziona il tipo di allenamento tra i seguenti:")
    for code, workout in workout_mapping.items():
        print(f"{code}: {workout}")
    workout_type = int(input("Inserisci il codice del tipo di allenamento: "))
    
    session_duration = float(input("Inserisci la durata della sessione in ore: "))
    weight = float(input("Inserisci il peso in kg: "))
    height = float(input("Inserisci l'altezza in metri: "))
    age = int(input("Inserisci l'età: "))
    avg_bpm = float(input("Inserisci la media dei battiti per minuto a fine allenamento: "))
    
    return gender, age, weight, height, avg_bpm, workout_type, session_duration

def scale_data(data, scaler):
    """
    Scala i dati forniti utilizzando lo scaler fornito.

    Parameters:
        data (DataFrame): i dati da scalare.
        scaler (Scaler): lo scaler da utilizzare per la normalizzazione.

    Returns:
        DataFrame: i dati scalati.
    """
    scaled_values = scaler.transform(data)
    return pd.DataFrame(scaled_values, columns=data.columns)

def make_predictions(models, scaled_data):
    """
    Effettua le predizioni sui dati scalati utilizzando i vari modelli di machine learning.

    Parameters:
        models (dict): i modelli di machine learning da utilizzare.
        scaled_data (DataFrame): i dati scalati sui quali fare le predizioni.

    Returns:
        dict: un dizionario con le predizioni per ogni modello e per l'intervallo di confidenza del modello Bayesian Ridge.
    """
    predictions = {
        "Regressione Lineare": models["linear"].predict(scaled_data)[0],
        "Random Forest": models["random_forest"].predict(scaled_data)[0],
        "Gradient Boosting": models["gradient_boosting"].predict(scaled_data)[0],
        "Bayesian Ridge": models["bayesian_ridge"].predict(scaled_data.to_numpy())[0]
    }
    
    mean_bayesian, std_bayesian = models["bayesian_ridge"].predict(scaled_data.to_numpy(), return_std=True)
    variance_bayesian = std_bayesian[0]
    conf_interval = 1.96 * variance_bayesian
    predictions["Intervallo di Confidenza Bayesian Ridge"] = (mean_bayesian[0] - conf_interval, mean_bayesian[0] + conf_interval)
    
    return predictions

def print_predictions(predictions):
    """
    Stampa i risultati delle predizioni effettuate.

    Parameters:
        predictions (dict): un dizionario contenente le predizioni per ogni modello.
    """
    print("\n=== Risultati delle Predizioni ===")
    for model, value in predictions.items():
        if isinstance(value, tuple):
            print(f"{model}: [{value[0]:.2f}, {value[1]:.2f}] kcal")
        else:
            print(f"{model}: {value:.2f} kcal")

def main():
    """
    Funzione principale che gestisce il flusso di lavoro dell'applicazione.

    1. Carica i modelli, lo scaler e la mappatura degli allenamenti.
    2. Raccoglie i dati dell'utente.
    3. Interroga la base di conoscenza Prolog per inferire il tipo di allenamento, l'intensità e la durata ottimale.
    4. Scala i dati dell'utente.
    5. Effettua le predizioni utilizzando i modelli di machine learning.
    6. Stampa i risultati delle predizioni.
    """
    models, scaler, workout_mapping = load_models()
    gender, age, weight, height, avg_bpm, workout_type, session_duration = get_user_input(workout_mapping)
    
    workout_type, intensity, optimal_duration = query_prolog(weight, height, session_duration)
    print_prolog_results(workout_mapping, workout_type, intensity, optimal_duration)
    
    user_data = pd.DataFrame([[gender, workout_type, session_duration, weight, age, avg_bpm]],
                              columns=['Gender', 'Workout_Type', 'Session_Duration_(hours)', 'Weight_(kg)', 'Age', 'Avg_BPM'])
    
    scaled_data = scale_data(user_data, scaler)
    predictions = make_predictions(models, scaled_data)
    print_predictions(predictions)
    input("\nPremi invio per uscire...")

if __name__ == "__main__":
    main()