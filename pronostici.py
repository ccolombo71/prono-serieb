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




X_encoded_home = pd.get_dummies(X['HomeTeam'], prefix='Home')
X_encoded_away = pd.get_dummies(X['AwayTeam'], prefix='Away')


# Concatena i DataFrame vettorizzati
X = pd.concat([X.drop(['HomeTeam', 'AwayTeam'], axis=1), X_encoded_home, X_encoded_away, merged_df['x_FTHG'], merged_df['x_FTAG'],merged_df.Sconfitte_c, merged_df.Sconfitte_f, merged_df.Pareggi_c, merged_df.Pareggi_f, merged_df.Vittorie_c, merged_df.Vittorie_f, merged_df['PA'], merged_df['PH']], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Valutazione del modello sui dati di training
p_train = model.predict(X_train)


# Valutazione del modello sui dati di test
p_test = model.predict(X_test)

# Plot confusion matrix per i dati di test
# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, p_test)

# Normalizza la matrice di confusione
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Crea un heatmap della matrice di confusione
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Vittoria fuoricasa", "Pareggio", 'Vittoria casa'], yticklabels=["Vittoria fuoricasa", "Pareggio", 'Vittoria casa'])
plt.xlabel('Predizione')
plt.ylabel('Reale')
plt.title('Matrice di Confusione')
st.pyplot(fig)
st.markdown(
    """
    **Interpretazione della Matrice di Confusione:**

    Ogni cella della matrice rappresenta una combinazione specifica di previsioni e risultati reali. 
    Gli elementi sulla diagonale principale rappresentano le previsioni corrette. 
    Gli elementi al di fuori della diagonale principale rappresentano gli errori di previsione.
    
    Ad esempio:

    **Vittoria squadra casa (Predizione) - Vittoria squadra casa (Reale):** Questa cella mostra la percentuale di partite dove il modello ha correttamente previsto che la squadra di casa avrebbe vinto, e questa è effettivamente la situazione in cui la squadra di casa ha vinto.
    """
)



