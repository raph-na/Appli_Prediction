from flask import Flask, render_template, request, flash, session, redirect, url_for
import os , uuid , base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO , BytesIO
from Pipeline.pretraitement import NettoyageAuto ,Nettoyer_Et_Gerer_date 
from Pipeline.pretraitement import build_future_features, detect_column_types
from Pipeline.pretraitement import temporal_train_test_split, compute_future_date, arrondi_par_exces
from Pipeline.histogramme import create_hist, create_heatmap_manquants, create_pairplot
from Pipeline.histogramme import create_heatmap, create_hist_temporals, create_scatter, create_hist_temporals_predict
from Pipeline.model import modelisation_avec_Random_Forest
from Pipeline.model import modelisation_avec_HBGR
from Pipeline.model import resultat_model
from Pipeline.encoder import EncodageAuto


app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici'  # IMPORTANT pour flash messages

UPLOAD_FOLDER = "uploads"
DATASETS_FOLDER = "datasets"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Créer les dossiers nécessaires
for folder in [UPLOAD_FOLDER, DATASETS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

#permet d'injecter des commandes pandas diectement dans la page
def capture_df_info(df):
    """Capture la sortie de df.info() dans une chaîne"""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# ===============================
# Première route
# ===============================
@app.route('/')
def home():
    return render_template('accueil.html')


# ===============================
# Page importation
# ===============================
@app.route('/gestion_importation')
def gestion_importation():
    return render_template('importation.html', fichier_upload=False)

# ===============================
# Page acceuil
# ===============================
@app.route('/gestion_acceuil')
def gestion_acceuil():
    return render_template('accueil.html')


# ===============================
# Page prediction
# ===============================
@app.route('/gestion_prediction')
def gestion_prediction():
    return render_template('prediction.html', fichier_upload=False)


# ===============================
# fonction de chargement
# ===============================
def Load(f):
    if not f or f.filename == '':
         flash("Aucun fichier selectionner")
         return None
    # Vérifier l'extension du fichier
    allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
    file_ext = os.path.splitext(f.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        flash("Format non supporté. Utilisez CSV, Excel ou JSON")
        return None
    # Sauvegarde temporaire
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(filepath)
    # Charger selon l'extension
    if file_ext == '.csv':
        df = pd.read_csv(filepath)
    elif file_ext in ['.xlsx', '.xls']:
         df = pd.read_excel(filepath)
    elif file_ext == '.json':
        df = pd.read_json(filepath)
    
    df = detect_column_types(df)
            # Nettoyer le fichier temporaire
    #if os.path.exists(filepath):
     #   os.remove(filepath)
    return df


def Load_from_filename(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    if not os.path.exists(filepath):
         return None
    # Charger selon l'extension
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
         df = pd.read_excel(filepath)
    elif filename.endswith('.json'):
          df = pd.read_json(filepath)
    return df

def save_from_nettoyage(df,name):
                original_name = session['filename']
                new_name = original_name.replace('.', '_'+name+'.')

                filepath = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
                    # stocke selon l'extension
                if original_name.endswith('.csv'):
                    df.to_csv(filepath, index = False)
                elif original_name.endswith('.xlsx') or original_name.endswith('.xls'):
                    df.to_excel(filepath, index = False)
                elif original_name.endswith('.json'):
                    df.to_json(filepath, index = False)

                session['filename'] = new_name
                return df




def Analyse_des_données(df):
    """Prépare toutes les données pour l'affichage de l'analyse"""
    # Aperçu du DataFrame
    headt = df.head().to_html(index=False, border=0, classes="dataframe")
    tailt = df.tail().to_html(index=False, border=0, classes="dataframe")
    
    # Informations sur le DataFrame
    infot = capture_df_info(df)
    
    # Statistiques descriptives
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        describet = df.describe().to_html(border=0, classes="dataframe")
    else:
        describet = "Aucune colonne numérique"
    
    # Valeurs manquantes
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        isnullt = df.isnull().sum().to_frame('Valeurs manquantes').to_html(border=0, classes="dataframe")
        isnull_sumt = int(missing_count)
    else:
        isnullt = "<p>Aucune valeur manquante</p>"
        isnull_sumt = 0
    
    # Informations générales
    shape_text = f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes"
    columns_list = list(df.columns)
    doublon = int(df.duplicated().sum())
    
    # Colonnes par type
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        'head': headt,
        'tail': tailt,
        'info': infot,
        'describe': describet,
        'isnull': isnullt,
        'isnull_sum': isnull_sumt,
        'shape': shape_text,
        'columns': columns_list,
        'doublons': doublon,
        'numeric_cols': numeric_columns,
        'categ_cols': categorical_columns,
        'row_count': df.shape[0],
        'col_count': df.shape[1],
        'plot_url'  : create_heatmap(df),
        'histogram_urls' : create_hist(df),
        'histogram_temporals': create_hist_temporals(df , session.get("date")),
        'heatmap_manquants': create_heatmap_manquants(df),
        'pairplot': create_pairplot(df)
    }



def Analyse_preliminaire(df):
    """Prépare toutes les données pour l'affichage de l'analyse"""
    # Aperçu du DataFrame
    headt = df.head().to_html(index=False, border=0, classes="dataframe")
    tailt = df.tail().to_html(index=False, border=0, classes="dataframe")
    
    # Informations sur le DataFrame
    infot = capture_df_info(df)
    
    # Statistiques descriptives
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        describet = df.describe().to_html(border=0, classes="dataframe")
    else:
        describet = "Aucune colonne numérique"
    
    # Valeurs manquantes
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        isnullt = df.isnull().sum().to_frame('Valeurs manquantes').to_html(border=0, classes="dataframe")
        isnull_sumt = int(missing_count)
    else:
        isnullt = "<p>Aucune valeur manquante</p>"
        isnull_sumt = 0
    
    # Informations générales
    shape_text = f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes"
    columns_list = list(df.columns)
    doublon = int(df.duplicated().sum())
    
    # Colonnes par type
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        'head': headt,
        'tail': tailt,
        'info': infot,
        'describe': describet,
        'isnull': isnullt,
        'isnull_sum': isnull_sumt,
        'shape': shape_text,
        'columns': columns_list,
        'doublons': doublon,
        'numeric_cols': numeric_columns,
        'categ_cols': categorical_columns,
        'row_count': df.shape[0],
        'col_count': df.shape[1]
    }


# ===============================
#  Route nettoyer
# ===============================
@app.route("/nettoyer" ,methods=['POST'])  
def nettoyer():
            df = Load_from_filename(session['filename'])
            if df is None:
                 flash("Aucun fichier analyser, charger d'abord un fichier")
                 return redirect("/gestion_importation")
            try:
                df = NettoyageAuto(df)
                flash("nettoyage appliqué avec succès")
                df = save_from_nettoyage(df,"clean")
                analysis_data = Analyse_des_données(df)
                return render_template('importation.html',
                                    fichier_upload=True,
                                    filename=session['filename'],
                                    date_column = session.get("date"),
                                    **analysis_data)
            except Exception as e:
                 flash(f"erreur lors du nettoyage: {str(e)}")
                 return redirect("/gestion_importation")
                 
# ===============================
#  Route encoder
# ===============================
@app.route("/encoder" ,methods=['POST'])  
def encoder():
            df = Load_from_filename(session['filename'])
            if df is None:
                 flash("Aucun fichier analyser, charger d'abord un fichier")
                 return redirect("/gestion_importation")
            try:
                df = EncodageAuto(df)
                flash("encodage appliqué avec succès")
                df = save_from_nettoyage(df,"encode")
                analysis_data = Analyse_des_données(df)
                return render_template('importation.html',
                                    fichier_upload=True,
                                    filename=session['filename'],
                                    date_column = session.get("date"),
                                    **analysis_data)
            except Exception as e:
                 flash(f"erreur lors de l'encodage: {str(e)}")
                 return redirect("/gestion_importation") 



# ===============================
#  Route importer  de la page importation
# ===============================
@app.route("/importer" ,methods=['POST'])  
def importer():

    if request.method=="POST":
            f = request.files.get("fichier") 
            if not f:
                  flash("Aucun fichier sélectionné")
                  return redirect("/gestion_importation")
            #chargement
            df = Load(f)
            session['filename'] = f.filename
            analysis_data = Analyse_preliminaire(df)
            return render_template('importation.html',
                                    fichier_upload=True,
                                    filename=session.get("filename"),
                                    **analysis_data)

# ===============================
#  Route Analyser
# ===============================
@app.route("/Analyser" ,methods=['POST'])  
def Analyser():
            
            df = Load_from_filename(session['filename'])
            date_column = request.form.get('date')
            session['date'] = date_column
            if df is None:
                 flash("Aucun fichier analyser, charger d'abord un fichier")
                 return redirect("/gestion_importation")
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                df = df.sort_values(date_column)
                df = df.set_index(date_column)

                analysis_data = Analyse_des_données(df)
                return render_template('importation.html' , 
                                   fichier_upload=True,
                                   filename=session.get("filename"),
                                   date_column = date_column,
                                   **analysis_data)
            except Exception as e:
                 flash(f"erreur lors du Traitement: {str(e)}")
                 return redirect("/gestion_importation") 





def Resultats_des_predictions(df):
    # Informations générales
    shape_text = f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes"
    columns_list = list(df.columns)
    # Valeurs manquantes
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        isnull_sumt = int(missing_count)
    else:
        isnull_sumt = 0
    doublon = int(df.duplicated().sum())
    
    # Colonnes par type
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        'isnull_sum': isnull_sumt,
        'shape': shape_text,
        'columns': columns_list,
        'doublons': doublon,
        'numeric_cols': numeric_columns,
        'row_count': df.shape[0],
        'col_count': df.shape[1]     
    }


# ===============================
#  Route import_prediction
# ===============================
@app.route("/import_prediction" ,methods=['POST'])  
def import_prediction():
     if request.method=="POST":
            f = request.files.get("monfichier") 
            if not f:
                  flash("Aucun fichier sélectionné")
                  return redirect("/gestion_prediction")
            #chargement
            df = Load(f)
            session['filename'] = f.filename
            predictys_data = Resultats_des_predictions(df)
            return render_template('prediction.html' , 
                                   fichier_upload=True,
                                   filename=session.get("filename"),
                                   **predictys_data)



# ===============================
#  Route pour choisir la target et la colone date
#  Et pour faire la prediction
# ===============================
@app.route('/choix_target', methods=['POST'])
def choix_target():

    # Si le formulaire est soumis
    if request.method == 'POST':
        df = Load_from_filename(session['filename'])
        df_copie = df
        if df is None:
                flash("Aucun fichier analyser, charger d'abord un fichier")
                return render_template('prediction.html' , fichier_upload=False) 
        
        #recuperation des choix des dates
        n_days = int(request.form.get('n_days', 0))
        n_months = int(request.form.get('n_months', 0))
        n_years = int(request.form.get('n_years', 0))  
        # validation obligatoire
        selections = [n_days > 0, n_months > 0, n_years > 0]
        if sum(selections) != 1:
            flash("Erreur : choisissez soit jours, soit mois, soit années.")
            return redirect("/gestion_prediction")
        
        # recuperation de la target et de la feature date
        target_column = request.form.get('target')
        date_column = request.form.get('date')
        if not target_column:
            flash("Erreur : Veuillez selectionner une colone cible.")
            return redirect("/gestion_prediction")
        if not date_column:
            flash("Erreur : Veuillez selectionner la colone date.")
            return redirect("/gestion_prediction")
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        df = df.dropna(subset=[date_column])
        # creation de la date a prédire
        last_date = df[date_column].max()

        # Application des fonction du fichier pretraitement
        df, lags = Nettoyer_Et_Gerer_date(df,target_column, date_column)
        
        # Séparé les colones numerique et categorielle
        cat_cols = []
        num_cols = []
        for col in df.columns:
                if col == date_column:
                    continue
                if col == target_column:
                    continue
                if df[col].dtype =='object':
                    cat_cols.append(col)
                else:
                   num_cols.append(col)

        # forcer la conversion des colones categorielles en String            
        for col in cat_cols:
               df[col] =  df[col].astype(str)        
        
        # Si une target a été choisie, séparer X et Y
        if target_column:
            y = df[[target_column]]
            X = df.drop(columns=[target_column])
        X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, ratio=0.8)
        
        # choix du modèle
        modele_choisi = request.form.get("modele")
        if modele_choisi == "hbgr":
           model_best = modelisation_avec_HBGR(X_train, y_train ,num_cols, cat_cols)
        elif modele_choisi == "rf":
           model_best = modelisation_avec_Random_Forest(X_train, y_train ,num_cols, cat_cols)
        else:
           flash("Veuillez choisir le modèle a utilisé")
           return redirect("/gestion_prediction")
        
        # première prédiction et évaluation
        rmse, r2_Random, mae, y_pred = resultat_model(model_best, X_test, y_test)

        future_date = compute_future_date(last_date, n_days, n_months, n_years)
        X_future = build_future_features(df, future_date, target_column, lags)

         # Ajouter les variables categorielles             
        for col in cat_cols:
             X_future[col] =  df[col].iloc[-1]        
        
        X_future = X_future.reindex(columns=X_train.columns, fill_value=0)
        # Verifier que les tailles des colones sont les mm
        #X_future = X_future[X_train.columns]
        
        # Faire la prédiction
        y_pred_future = model_best.predict(X_future)


        # Affichage des resultats
        if rmse == None or r2_Random== None or mae == None:
            flash("Erreur : Aucune prediction généré")   
            return redirect("/gestion_prediction")
        
        
        
        Xt = X.head().to_html(index=False, border=0, classes="dataframe")
        yt = y.head().to_html(index=False, border=0, classes="dataframe")
        
    # Liste des colonnes (tu peux filtrer selon les colonnes pertinentes)
    #columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    predictys_data = Resultats_des_predictions(df)
    return render_template('prediction.html' , 
                                   fichier_upload=True,
                                   filename=session.get("filename"),
                                   target_column=target_column,
                                   date_column=date_column,
                                   X=Xt, y=yt, R2_SCORE=arrondi_par_exces(r2_Random), xf=X_future,
                                   MAE= arrondi_par_exces(mae), RMSE = arrondi_par_exces(rmse) ,
                                   scatter_urls = create_scatter(X, y,target_column, y_pred),
                                   temporals_predict = create_hist_temporals_predict(df_copie , date_column),
                                   y_pred = float(y_pred_future[0]),
                                   **predictys_data)



if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=True)