from pathlib import Path

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import plotly.express as px

# Root path
ROOT_PATH = Path(__file__).resolve().parents[2]

# Data paths
DATA_PATH = ROOT_PATH / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw'
AUGMENTED_DATA_PATH = DATA_PATH / 'augmented'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
DATA_VISUALIZATION_PATH = DATA_PATH / 'visualization'
FEATURES_PATH = DATA_PATH / 'features'

# Results paths
RESULTS_DIR = ROOT_PATH / 'results'

# Sets percentage of samples
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# Styles for dash
styles = {
    'container': {
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
        'minHeight': '100vh',
        'background': 'linear-gradient(to right, #4b6cb7, #182848)',
        'fontFamily': 'Arial, sans-serif',
        'padding': '20px',
        'boxSizing': 'border-box'
    },
    'title': {
        'textAlign': 'center',
        'color': 'white',
        'fontSize': '2rem',
        'marginBottom': '20px',
        'marginTop': '20px',
        'fontWeight': 'bold'
    },
    'body': {
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'width': '90%',
        'maxWidth': '2000px',
        'backgroundColor': 'white',
        'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.3)',
        'borderRadius': '10px',
        'padding': '20px',
        'boxSizing': 'border-box'
    },
    'graficas': {
        'width': '100%',
        'height': '1000px',
        'marginBottom': '10px'
    },
    'controls': {
        'display': 'flex',
        'flexDirection': 'row',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'gap': '10px',
        'padding': '10px',
        'background': '#f5f5f5',
        'borderRadius': '8px',
        'boxShadow': '0px 2px 5px rgba(0, 0, 0, 0.2)',
        'width': '100%',
        'boxSizing': 'border-box'
    },
    'slider-container': {
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'textAlign': 'center',
        'width': '50px',
        'height': '300px',
        'marginLeft': '5px',
        'marginRight': '5px',
    },
    'slider-label': {
        'fontSize': '0.9rem',
        'marginBottom': '5px',
        'color': '#333',
        'fontWeight': 'bold',
        'textAlign': 'center',
    }
}

# Graphing parameters
offset = 2
colors = {
    'Originals': px.colors.qualitative.Plotly[0],
    'LATs': px.colors.qualitative.Plotly[1],
    'Binary': px.colors.qualitative.Plotly[2],
    'Discrete': px.colors.qualitative.Plotly[3],
    'Fragmented': px.colors.qualitative.Plotly[4],
    'Entanglement': px.colors.qualitative.Plotly[5],
    'Reconstructed': 'red',
    'Modified': px.colors.qualitative.Plotly[7]
}

colors_dims = {
    'dim_0': px.colors.qualitative.Plotly[0],
    'dim_1': px.colors.qualitative.Plotly[1],
    'dim_2': px.colors.qualitative.Plotly[2],
    'dim_3': px.colors.qualitative.Plotly[3],
    'dim_4': px.colors.qualitative.Plotly[4],
    'dim_5': px.colors.qualitative.Plotly[5],
    'dim_6': px.colors.qualitative.Plotly[6],
    'dim_7': px.colors.qualitative.Plotly[7]
}

# Testing autoencoders architectures
layers = [[8, 8], [8, 16], [16, 16], [16, 32], [32, 32], [32, 64], [64, 64]]
lds = [8, 16, 64]

# Classifiers parameters
seed = 42
classifiers = ['RF', 'Gradient Boosting', 'SVM', 'LR', 'KNN', 'XGBoost', 'LightGBM', 'CatBoost']
models_params = {
    "LR": (LogisticRegression(random_state=seed), {
        'C': [0.1, 1, 10]
    }),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 11, 13, 15]
    }),
    "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=seed), {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9]
    }),
    "LightGBM": (LGBMClassifier(random_state=seed), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    "CatBoost": (CatBoostClassifier(verbose=False, random_state=seed), {
        'depth': [4, 6, 8],
        'iterations': [100, 200]
    })
}
