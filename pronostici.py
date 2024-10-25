import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import scikitplot as skplt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import streamlit as st
import seaborn as sns
from sklearn.metrics import confusion_matrix
import requests
from bs4 import BeautifulSoup
import pandas as pd

st.set_page_config(page_title="Pronostici con il Machine Learning")
st.title('Pronostici partite SERIE B con il Machine Learning')
st.header('Pronostici partite per la stagione in corso')
st.write(
    "Benvenuto all'applicazione Pronostici partite con il Machine Learning. "
    "Questa applicazione utilizza modelli di Machine Learning per predire gli esiti delle partite di calcio "
    "Scorri verso il basso per vedere le previsioni delle prossime partite e la classifica prevista."
)
np.random.seed(2)

def converti_data(data_str):
    try:
        return pd.to_datetime(data_str, format='%d/%m/%Y')
    except ValueError:
        
        try:
            return pd.to_datetime(data_str, format='%d/%m/%y')
        except ValueError:
            print(f"Formato data non riconosciuto per: {data_str}")
            return None 


# Carica i dati da ciascun URL in un DataFrame
def load_dataframes():
    dataframes = {}
    for year in range(5, 24):
        url = f'https://www.football-data.co.uk/mmz4281/{year:02d}{year + 1:02d}/I2.csv'
        df_name = f'df{year:02d}'
        globals()[df_name] = pd.read_csv(url)
        dataframes[df_name] = globals()[df_name]

dataframes = load_dataframes()


dfs = [df23, df22, df21, df20, df19, df18, df17, df16, df15, df14, df13, df12, df11, df10, df09, df08, df07, df06, df05]  # Aggiungi tutti i tuoi DataFrame qui



for df in dfs:
    df['Date'] = df['Date'].apply(converti_data)
    df.sort_values(by='Date', inplace=True)
    df['x_FTHG'] = df.groupby('HomeTeam')['FTHG'].cumsum() - df['FTHG']
    df['x_FTAG'] = df.groupby('AwayTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTHGS'] = df.groupby('HomeTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTAGS'] = df.groupby('AwayTeam')['FTHG'].cumsum() - df['FTHG']
    df['x_FTHG_R'] = df.groupby('HomeTeam')['FTHG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTHG']
    df['x_FTAG_R'] = df.groupby('AwayTeam')['FTAG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['FTAG']
    df['x_FTHGS_R'] = df.groupby('HomeTeam')['FTAG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTAG']
    df['x_FTAGS_R'] = df.groupby('AwayTeam')['FTHG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTHG']
    df['x_HS'] = df.groupby('HomeTeam')['HS'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HS']
    df['x_AS'] = df.groupby('AwayTeam')['AS'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AS']
    df['x_HST'] = df.groupby('HomeTeam')['HST'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HST']
    df['x_AST'] = df.groupby('AwayTeam')['AST'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AST']
    df['x_HC'] = df.groupby('HomeTeam')['HC'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HC']
    df['x_AC'] = df.groupby('AwayTeam')['AC'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AC']
    df['x_HF'] = df.groupby('HomeTeam')['HF'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HF']
    df['x_AF'] = df.groupby('AwayTeam')['AF'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AF']
    df['conta']=1
    df['PGH'] = df.groupby('HomeTeam')['conta'].cumsum() - df['conta']
    df['PGA'] = df.groupby('AwayTeam')['conta'].cumsum() - df['conta']
    # Ottieni le colonne univoche dalla colonna 'risultato'
    lettere_univoche = df['FTR'].unique()
    # Per ogni lettera unica, crea una nuova colonna nel DataFrame
    for lettera in lettere_univoche:
        df[f'{lettera}_home'] = (df['FTR'] == lettera).astype(int)

    df['Sconfitte_c'] = df.groupby('HomeTeam')['A_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['A_home']
    df['Pareggi_c'] = df.groupby('HomeTeam')['D_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['D_home']
    df['Vittorie_c'] = df.groupby('HomeTeam')['H_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['H_home']
    for lettera in lettere_univoche:
        df[f'{lettera}_away'] = (df['FTR'] == lettera).astype(int)
    df['Pareggi_f'] = df.groupby('AwayTeam')['D_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['D_away']
    df['Vittorie_f'] = df.groupby('AwayTeam')['A_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['A_away']
    df['Sconfitte_f'] = df.groupby('AwayTeam')['H_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['H_away']
    mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
    df['PH'] = df['FTR'].map(mappatura_valori_h)
    df['PH']=df.groupby('HomeTeam')['PH'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True) - df['PH']
    mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
    df['PA'] = df['FTR'].map(mappatura_valori_a)
    df['PA'] = df.groupby('AwayTeam')['PA'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True) - df['PA']



# Unisci i DataFrame in un unico DataFrame
merged_df = pd.concat([df23,df22, df21, df20, df19, df18, df17, df16, df15, df14, df13, df12, df11, df10, df09, df08, df07, df06, df05], ignore_index=True)

merged_df = merged_df.dropna(subset=['AwayTeam'])

X = merged_df[['HomeTeam', 'AwayTeam']]
y = merged_df['FTR']


