import pandas as pd
import numpy as np
import math
import warnings
from datetime import timedelta
warnings.filterwarnings("ignore",message="could not infer format")
#from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# ====================================
#  2. Détection automatique des types
# ====================================
def detect_column_types(df):
    nbre=0
    print("Etape 2:  Détection automatique des types...")
    for col in df.columns:
        # Ignorer les colonnes vides
        if df[col].dropna().shape[0] == 0:
            continue
        
        # Tentative de conversion numérique
        sample = df[col].dropna().astype(str).str.replace(',', '.')
        if sample.str.replace('.', '', 1).str.isnumeric().mean() > 0.8:
            df[col] = pd.to_numeric(sample, errors='coerce')
            print(f"    {col} → numérique")
            nbre+=1

        # Tentative de conversion en date
        elif pd.to_datetime(sample, errors='coerce').notna().mean() > 0.8:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"   {col} → date")
            nbre+=1

        else:
            df[col] = df[col].astype(str)
            print(f"  {col} → texte (object)")
            nbre+=1
    print(f"         {nbre} Types détectés.")
    print(f" {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df



# ======================================
#  3. Imputation des valeurs manquantes
# ======================================
def impute_missing_values(df):
    print("Etape 3:  Imputation des valeurs manquantes...")
    for col in df.columns:
        #   Si la colonne est numérique (float ou int)
        if df[col].dtype in ['float', 'int']:
            if df[col].isna().any():
                df[col]=df[col].fillna(df[col].median())
                print(f" Imputation des valeurs de {col}.")
    
        #   Si la colonne est de type date/temps
        elif np.issubdtype(df[col].dtype, np.datetime64):
             df[col]=df[col].fillna(method='ffill')  # Remplit avec la valeur précédente
             print(f" Imputation des valeurs de {col}.")
    
        #  Si la colonne est catégorielle ou texte (object)
        elif df[col].dtype == 'object':
            if df[col].isna().any():  # Vérifie qu’il y a au moins une valeur manquante
                 df[col] = df[col].fillna(df[col].mode()[0])
                 print(f" Imputation des valeurs de {col}.")
    
        #  Sinon (par précaution)
        else:
            df[col]=df[col].fillna(method='ffill')
            print(f" Imputation des valeurs de {col}.")
    print(f" Imputation terminée.")
    return df

# ===============================
#  4. Suppression des doublons
# ===============================
def remove_duplicates(df):
    print("Etape 4: gestions des doublons")
    doublons=df.duplicated().sum()
    if(doublons>0):
        df=df.drop_duplicates().reset_index(drop=True)
        print(f"nombres de doublons {doublons} supprimé avec succès")
    else :
        print("aucun doublons doublons")
    return df


def NettoyageAuto(df):
    df = detect_column_types(df)
    df = impute_missing_values(df)
    df = remove_duplicates(df)
    return df


# =========================================================
#  gestions des dates pour pouvoir faire la prediction
# =========================================================
def gestion_dates(df, date_col):
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df =df.drop(df[[date_col]],axis=1)
            return df


def create_lags(df, target_col, freq):
    # Vérifier si freq est None ou non-string
    if freq is None:
        lags = [1, 2, 3]
    elif isinstance(freq, str):
        if freq.startswith('D'):
            lags = [1, 7, 14, 30]
        elif freq.startswith('M'):
            lags = [1, 3, 6, 12]
        elif freq.startswith('W'):
            lags = [1, 4, 12, 52]
        else:
            lags = [1, 2, 3]
    else:
        lags = [1, 2, 3]  # Valeur par défaut
    
    # Vérifier que la colonne target existe
    if target_col in df.columns:
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    else:
        print(f"Attention: Colonne cible '{target_col}' non trouvée pour créer les lags")
    
    return df, lags
 

#nettoyage apres choix utilisateur et detection de la frequence temporelle
def Nettoyer_Et_Gerer_date(df, target_col, date_col):
    df = NettoyageAuto(df)
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Détection de fréquence avec gestion des erreurs
    freq = pd.infer_freq(df[date_col])
    
    # Si la fréquence n'est pas détectée, la calculer manuellement
    if freq is None:
        print("Fréquence non détectée automatiquement, calcul manuel...")
        if len(df) > 1:
            # Calculer les différences entre dates consécutives
            date_diffs = df[date_col].diff().dropna()
            if not date_diffs.empty:
                # Prendre la différence la plus fréquente
                most_common_diff = date_diffs.mode()[0]
                # Convertir en fréquence pandas
                if pd.Timedelta(days=1) <= most_common_diff <= pd.Timedelta(days=2):
                    freq = 'D'  # Journalier
                elif pd.Timedelta(days=7) <= most_common_diff <= pd.Timedelta(days=31):
                    freq = 'W'  # Hebdomadaire
                elif pd.Timedelta(days=28) <= most_common_diff <= pd.Timedelta(days=32):
                    freq = 'M'  # Mensuel
                else:
                    freq = 'D'  # Par défaut journalier
        else:
            freq = 'D'  # Valeur par défaut
    
    print(f"Fréquence détectée: {freq}")
    
    # Resample seulement si on a une fréquence valide
    try:
        df = df.set_index(date_col).resample(freq).sum().reset_index()
        print(f"Resample possible avec fréquence Good")
    except Exception as e:
        print(f"Attention: Resample impossible avec fréquence {freq}: {str(e)}")
        print("Utilisation du dataframe sans resample...")
        # Garder les données telles quelles
    
    df = gestion_dates(df, date_col)
    df, lags = create_lags(df, target_col, freq)
    return df, lags
    



#train test split temporel
def temporal_train_test_split(X, y, ratio=0.8):
    split = int(len(X) * ratio)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test




  
  
# =========================================================
#  section pour faire la prediction
# =========================================================

# #calcul de la date future
def compute_future_date(last_date, n_days=0, n_months=0, n_years=0):
    if n_days > 0:
        return last_date + pd.Timedelta(days=n_days)
    if n_months > 0:
        return last_date + pd.DateOffset(months=n_months)
    if n_years > 0:
        return last_date + pd.DateOffset(years=n_years)
    

# construction de x future
def build_future_features(df, future_date, target_col, lags):
    last_row = df.iloc[-1]

    data = {
        'day': future_date.day,
        'year': future_date.year,
        'month': future_date.month,
        'month_sin': np.sin(2 * np.pi * future_date.month / 12),
        'month_cos': np.cos(2 * np.pi * future_date.month / 12),
    }

    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        if lag_col in df.columns:
           data[lag_col] = last_row[lag_col]
        else:
            data[lag_col] = last_row[target_col]
    return pd.DataFrame([data])
    
    
def arrondi_par_exces(nbre):
    return math.ceil(1000 *nbre) /1000