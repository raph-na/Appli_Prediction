import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore",message="could not infer format")
#from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# ===============================
#  6. Encodage automatique
# ===============================
def encode_categoricals(df):
    print("Etape 5:  Encodage des variables catégorielles...")
    for col in df.select_dtypes(include='object'):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            print(f" utilisation de Label Encoding sur {col}")
    print(" Encodage terminé.")
    #detection et remplacement des types True False
     #detection et remplacement des types True False
    for col in df.columns:
        # Ignorer les colonnes vides
        if df[col].dropna().isin([True,False,'True','False']).all():
             df[col] = df[col].astype(int)
 
    return df

def EncodageAuto(df):
    df = encode_categoricals(df)
    return df