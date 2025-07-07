import pandas as pd
from xgboost import XGBRegressor
import numpy as np

import utils.constants as cons
from utils.load_data import load_data, load_embeddings

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
batch_size = 64

aux_str_data = f'_L{int(window_size / subsampling)}_O{window_overlap}_S{2 * subsampling}.parquet'
train_data, val_data, test_data = load_data(type_signal, seed, window_size, window_overlap, subsampling,
                                            ['train', 'val', 'test'])
train_data, val_data, test_data = train_data.patient_id, val_data.patient_id, test_data.patient_id


# Load feature files for all patients in val and test sets
df_features_train = pd.DataFrame()
for patient in train_data.unique().tolist() + val_data.unique().tolist():
    features_path = (
            cons.AUGMENTED_DATA_PATH / 'features' /
            f'features_{patient}_CartoFinder_augmented{aux_str_data}'
    )
    if features_path.exists():
        patient_features = pd.read_parquet(features_path).drop(columns=['subsampling', 'DomCycleL']).fillna(0)
        df_features_train = pd.concat([df_features_train, patient_features])
    else:
        print(f'Features file {features_path} not found.')

df_features_test = pd.DataFrame()
for patient in test_data.unique().tolist():
    features_path = (
            cons.AUGMENTED_DATA_PATH / 'features' /
            f'features_{patient}_CartoFinder_augmented{aux_str_data}'
    )
    if features_path.exists():
        patient_features = pd.read_parquet(features_path).drop(columns=['subsampling', 'DomCycleL']).fillna(0)
        df_features_test = pd.concat([df_features_test, patient_features])
    else:
        print(f'Features file {features_path} not found.')

classifier = 'rf'

# ---------------------------
# Model parameters
# ---------------------------
kernel_size = 3
s = 1
padding = 1
pooling = True
norm = True
drop = 0.25

# ---------------------------
# AE architecture configurations
# ---------------------------
layers_channels = cons.layers
latent_dims = cons.lds

# Loop through all combinations of AE layer configurations and latent dimensions
for layer_channel in layers_channels:
    for latent_dim in latent_dims:

        # Build auxiliary strings for data and model identification
        aux_str_model = (
            f'ch{layer_channel}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}_ld{latent_dim}'
        )

        # Load patient metadata and embeddings for validation and test
        train_embeds, val_embeds, test_embeds = load_embeddings(type_signal, layer_channel, kernel_size, s, padding,
                                                                pooling, norm, drop, latent_dim,
                                                                ['train', 'val', 'test'])
        train_embeds, val_embeds, test_embeds = (pd.DataFrame(train_embeds), pd.DataFrame(val_embeds),
                                                 pd.DataFrame(test_embeds))
        train_embeds = pd.concat([train_embeds, val_embeds]).reset_index(drop=True)

        # Prepare directory to save regression scores
        scores_saving_path = cons.RESULTS_DIR / type_signal / aux_str_model / 'rf_regression'
        scores_saving_path.mkdir(parents=True, exist_ok=True)

        # Define which feature aggregations to analyze
        r2_scores = {}
        types_features = ['mean']

        # Perform regression for each type of summary feature
        for i, type_features in enumerate(types_features):
            if type_features == 'mean':
                patient_features_train = df_features_train.groupby(['patient_id', 'map_name', 'site_id', 'window'],
                                                                   sort=False).mean()
                patient_features_test = df_features_test.groupby(['patient_id', 'map_name', 'site_id', 'window'],
                                                                 sort=False).mean()
            else:
                patient_features_train = None
                patient_features_test = None

            # Reset index to match embeddings
            patient_features_train = patient_features_train.reset_index(drop=True)
            patient_features_test = patient_features_test.reset_index(drop=True)

            if classifier == 'svr':
                indices = np.linspace(0, len(patient_features_train) - 1, 10000, dtype=int)
                patient_features_train = patient_features_train.iloc[indices]
                train_embeds = train_embeds.iloc[indices]
                scaler = StandardScaler()
                train_embeds = scaler.fit_transform(train_embeds)
                test_embeds = scaler.transform(test_embeds)

            for target_col in patient_features_train.columns:
                y_train = patient_features_train[target_col]
                y_test = patient_features_test[target_col]

                if classifier == 'rf':
                    # Definimos el modelo base
                    model = XGBRegressor(random_state=42)

                    # Definimos los hiperparámetros que queremos buscar
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 1],
                    }

                    n_iter = 30

                elif classifier == 'svr':
                    # Definimos el modelo base
                    model = SVR()

                    # Definimos los hiperparámetros a buscar
                    param_grid = {
                        'C': [0.1, 1, 10, 100],
                        'epsilon': [0.01, 0.1, 0.2],
                        'kernel': ['rbf', 'linear'],
                    }

                    n_iter = 12
                else:
                    raise NotImplementedError

                # Configuramos el GridSearchCV
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,  # número de combinaciones aleatorias a probar
                    scoring='r2',  # métrica objetivo
                    cv=3,  # 5-fold cross-validation
                    verbose=2,
                    random_state=42,
                    n_jobs=-1
                )

                # Ejecutamos la búsqueda
                random_search.fit(train_embeds, y_train)

                # Resultados
                print(f"Mejores hiperparámetros: {random_search.best_params_}")
                print(f"Mejor score de CV (r2): {random_search.best_score_}")

                # Opcional: evaluamos en el conjunto de test
                best_model = random_search.best_estimator_
                y_pred = best_model.predict(test_embeds)
                r2 = r2_score(y_test, y_pred)
                print(f"R2 en test: {r2}")
                r2_scores[target_col] = r2

        # Save summary feature regression results
        r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R2_score']).sort_values(by='R2_score', ascending=False)
        r2_df.to_csv(scores_saving_path/f'{classifier}_r2_scores_rs', index=True)
