import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path):
    """
    Carica un dataset da un file CSV, prepara i dati per l'analisi e salva la mappatura dei tipi di allenamento.
    
    Parametri:
    - file_path (str): il percorso del file CSV contenente il dataset.
    
    Restituisce:
    - df (DataFrame): il dataset preparato con le variabili trasformate.
    """
    df = pd.read_csv(file_path)

    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    df['Workout_Type'] = df['Workout_Type'].astype('category')
    workout_mapping = dict(enumerate(df['Workout_Type'].cat.categories))
    joblib.dump(workout_mapping, 'modelli/workout_mapping.pkl') 

    df['Workout_Type'] = df['Workout_Type'].cat.codes

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    return df


def standardize_features(X_train, X_test, scaler_path="modelli/scaler.pkl"):
    """
    Applica la standardizzazione alle feature numeriche del dataset per prepararli all'addestramento del modello.
    
    Parametri:
    - X_train (DataFrame): il set di dati di addestramento contenente le feature numeriche.
    - X_test (DataFrame): il set di dati di test contenente le feature numeriche.
    - scaler_path (str): percorso dove salvare lo scaler (predefinito: 'modelli/scaler.pkl').
    
    Restituisce:
    - X_train_scaled (DataFrame): il set di addestramento con le feature standardizzate.
    - X_test_scaled (DataFrame): il set di test con le feature standardizzate.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path) 

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)