import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Pronostici con il Machine Learning")
st.title('Pronostici partite SERIE B con il Machine Learning')
st.header('Pronostici partite per la stagione in corso')
st.write(
    "Benvenuto all'applicazione Pronostici partite con il Machine Learning. "
    "Questa applicazione utilizza modelli di Machine Learning per predire gli esiti delle partite di calcio. "
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
        
        # Carica il CSV specificando l'encoding e gestendo gli errori
        try:
            globals()[df_name] = pd.read_csv(url, encoding='ISO-8859-1')  # Specifica l'encoding qui
            dataframes[df_name] = globals()[df_name]
        except UnicodeDecodeError as e:
            print(f"Errore di decoding per {url}: {e}")
    
    return dataframes

# Carica tutti i DataFrame
dataframes = load_dataframes()

# Specifica i DataFrame da utilizzare successivamente
dfs = [dataframes.get(f'df{year:02d}', pd.DataFrame()) for year in range(23, 4, -1)]  # Modifica qui per usare i DataFrame caricati



