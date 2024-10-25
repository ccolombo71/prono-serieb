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






