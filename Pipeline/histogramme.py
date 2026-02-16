import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Pipeline.pretraitement import NettoyageAuto ,Nettoyer_Et_Gerer_date
from Pipeline.pretraitement import temporal_train_test_split, create_lags
from io import StringIO , BytesIO
from Pipeline.encoder import EncodageAuto
import os , uuid , base64
import io , gc

# Definição das cores base em tons de café/marrom para consistência visual
COLOR_MARROM_ESCURO = "#6F4E37"
COLOR_MARROM_MEDIO = "#A0522D"
COLOR_BEGE = "#D2B48C"

BASE_COLORS = [COLOR_MARROM_ESCURO, COLOR_MARROM_MEDIO, COLOR_BEGE]
COFFEE_CMAP = sns.light_palette(COLOR_MARROM_ESCURO, as_cmap=True)

def create_heatmap(df):
    # Calculer la matrice de corrélation
    df_enoded = EncodageAuto(df)
    corr = df_enoded.corr()

    # Créer une heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Carte thermique des corrélations')

    # Sauvegarder l'image dans un objet BytesIO
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)

    # Convertir l'image en base64 pour l'inclure dans une page HTML
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    del fig ,ax 
    gc.collect()
    return img_b64


def create_heatmap_manquants(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.isna(), cbar=True, yticklabels=False, cmap='viridis',ax=ax)
    ax.set_title('visualisation des manquants')
    # Sauvegarder l'image dans un objet BytesIO
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)

    # Convertir l'image en base64 pour l'inclure dans une page HTML
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    del fig ,ax 
    gc.collect()
    return img_b64

def create_pairplot(df):
    sns.pairplot(df)
    plt.title('visualisation des manquants')
    # Sauvegarder l'image dans un objet BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64 pour l'inclure dans une page HTML
    img_b63 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return img_b63


def create_hist(df):
    df = EncodageAuto(df)
         # Liste pour stocker les images en base64
    histogram_urls = []
    
    # Pour chaque colonne numérique, créer un histogramme
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Créer un histogramme
        sns.histplot(df[column], kde=True, color='blue', bins=20, ax=ax)
        ax.set_title(f'Distribution de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Fréquence')
        
        # Sauvegarder l'histogramme dans un objet BytesIO
        img = BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        
        # Convertir l'image en base64
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')     
        # Ajouter l'URL de l'histogramme à la liste
        histogram_urls.append(img_b64)      
        # Fermer le plot pour libérer la mémoire
        plt.close(fig)
    
    return histogram_urls


def create_hist_temporals(df , date_column):
    df = EncodageAuto(df)
         # Liste pour stocker les images en base64
    histogram_temporals = []
    # Pour chaque colonne numérique, créer un histogramme
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        
        # Créer un histogramme
        df[column].plot(figsize=(8, 6))
        plt.title(f'Distribution de {column}')
        plt.xlabel(date_column)
        plt.ylabel(column)
        plt.show
        # Sauvegarder l'histogramme dans un objet BytesIO
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Convertir l'image en base64
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        # Ajouter l'URL de l'histogramme à la liste
        histogram_temporals.append(img_b64)
        
        # Fermer le plot pour libérer la mémoire
        plt.close()
    
    return histogram_temporals

# =========================================================
#  histogramme de la partie prediction
# =========================================================


def create_scatter(x, y,target_column, y_pred):
    x = EncodageAuto(x)
         # Liste pour stocker les images en base64
    scatter_urls = []
    
    # Pour chaque colonne numérique, créer un histogramme
    for column in x.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Visualisation
        ax.scatter(x[column], y, color='blue')
        ax.set_title(f"Régression Linéaire Apprise {column} vs {target_column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        
        # Sauvegarder l'histogramme dans un objet BytesIO
        img = BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        
        # Convertir l'image en base64
        img_b62 = base64.b64encode(img.getvalue()).decode('utf-8')     
        # Ajouter l'URL de l'histogramme à la liste
        scatter_urls.append(img_b62)      
        # Fermer le plot pour libérer la mémoire
        plt.close(fig)
    
    return scatter_urls


def create_hist_temporals_predict(df , date_column):
    df_sorted = EncodageAuto(df)
         
    # Si plus de 200 points, échantillonner
    if len(df_sorted) > 200:
        # Garder le premier, le dernier, et un échantillon régulier
        step = len(df_sorted) // 200
        df_plot = df_sorted.iloc[::step].copy()
        # S'assurer que le dernier point est inclus
        if df_plot.iloc[-1].name != df_sorted.iloc[-1].name:
            df_plot = pd.concat([df_plot, df_sorted.iloc[[-1]]])
    else:
        df_plot = df_sorted

    # Liste pour stocker les images en base64
    histogram_temporals = []
    # Pour chaque colonne numérique, créer un histogramme
    for column in df.select_dtypes(include=['float64', 'int64']).columns:

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle(f'ANALISE TEMPOREL - EVOLUTION DES VENTES DE {column}', 
             fontsize=18, fontweight='bold')
        
        # Subplot 1: Receita Diária
        ax.plot(df[date_column], df[column], linestyle='-',
            color=COLOR_MARROM_ESCURO, linewidth=2, marker='', markersize=4 , label="Tendance")
        if len(df_plot) > 100:
        # Pas de remplissage
            pass
        else:
            ax.fill_between(df_plot[date_column], df_plot[column], 
                            alpha=0.2, color="#D2B48C")
        # Créer un histogramme
  

        ax.set_xlabel(date_column)
        ax.set_ylabel(column)
        ax.grid(True)
  
        # Sauvegarder l'histogramme dans un objet BytesIO
        img = BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        
        # Convertir l'image en base64
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')      
        # Ajouter l'URL de l'histogramme à la liste
        histogram_temporals.append(img_b64)
        # Fermer le plot pour libérer la mémoire
        plt.close(fig)
    
    return histogram_temporals

# Criar figura com 3 subplots


