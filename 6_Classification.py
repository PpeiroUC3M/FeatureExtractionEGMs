import os
import pickle
import numpy as np
import pandas as pd
import utils.constants as cons
from utils.load_data import load_embeddings, load_labels

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

vae = False

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
labels = 'focal'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42

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

# ---------------------------
# Classification parameters
# ---------------------------
save_results = True
save_model = True
balancing = 'Undersample'
classifiers = ['LR', 'KNN', 'XGBoost', 'LightGBM', 'CatBoost']
# classifiers = cons.classifiers

# Loop through each latent dimension and layer configuration
for latent_dim in latent_dims:
    for layer_channel in layers_channels:

        # Load embeddings for train, validation, and test sets
        train_embeds, val_embeds, test_embeds = load_embeddings(
            type_signal, layer_channel, kernel_size, s, padding,
            pooling, norm, drop, latent_dim, ['train', 'val', 'test'], vae
        )

        # Concatenate train and validation embeddings for final training set
        train_embeds = np.concatenate([train_embeds, val_embeds], axis=0)

        # Standardize embeddings using training data
        # scaler = StandardScaler()
        # train_embeds_scaled = scaler.fit_transform(train_embeds)
        # test_embeds_scaled = scaler.transform(test_embeds)

        # Load and combine labels for training (train + val) and test
        y_train, y_val, y_test = load_labels(type_signal, labels, seed, window_size, window_overlap, subsampling)
        y_train = np.concatenate([y_train, y_val])
        y_test = y_test.ravel()

        # Apply selected class balancing method
        if balancing == 'Undersample':
            sampler = RandomUnderSampler(random_state=42)
            train_embeds, y_train = sampler.fit_resample(train_embeds, y_train)
        elif balancing == 'SMOTE':
            smote = SMOTE(random_state=0)
            train_embeds, y_train = smote.fit_resample(train_embeds, y_train)
        else:
            raise ValueError('Unrecognized balancing method')

        # Standardize embeddings using training data
        scaler = StandardScaler()
        train_embeds_scaled = scaler.fit_transform(train_embeds)
        test_embeds_scaled = scaler.transform(test_embeds)

        # Use test set as-is
        x_train, y_train = train_embeds_scaled, y_train
        x_test, y_test = test_embeds_scaled, y_test

        metrics_results = []

        # Build string for identifying model directory
        aux_str_ae = (
            f'ch{layer_channel}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}_ld{latent_dim}_mse'
        )

        # Define paths for saving models and metrics
        if vae:
            models_saving_path = cons.RESULTS_DIR / 'vae' / type_signal / aux_str_ae / 'model'
            metrics_saving_path = (
                cons.RESULTS_DIR / 'vae' / type_signal / aux_str_ae / 'metrics' /
                f"model_evaluation_metrics_{labels}_{balancing}.csv"
            )
        else:
            models_saving_path = cons.RESULTS_DIR / type_signal / aux_str_ae / 'model'
            metrics_saving_path = (
                cons.RESULTS_DIR / type_signal / aux_str_ae / 'metrics' /
                f"model_evaluation_metrics_{labels}_{balancing}.csv"
            )

        # Train and evaluate each classifier using GridSearchCV
        best_models = {}
        for model in classifiers:
            print(f"Executing GridSearchCV for {model}...")

            # Grid search for best hyperparameters using cross-validation
            grid_search = GridSearchCV(
                cons.models_params[model][0], cons.models_params[model][1],
                cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_

            # Save model if flag is set
            best_models.update({model: best_model})

            # Predict on test set
            y_pred = best_model.predict(x_test)
            y_pred_proba = best_model.predict_proba(x_test)[:, 1]

            # Compute performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # Store metrics
            metrics_results.append({
                'Model': model,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'F1-Score': f1,
                'AUC': auc,
                'cm': cm,
                'hyperparams': grid_search.best_params_
            })

            # Print results
            print(cm)
            print(classification_report(y_test, y_pred))
            print(f"Best model for {model}: {grid_search.best_params_}")
            print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, "
                  f"F1-Score: {f1:.4f}, AUC: {auc:.4f}")

        # Save results to CSV, merge if file exists
        df_results = pd.DataFrame(metrics_results).sort_values(by=['Model'])
        if os.path.exists(metrics_saving_path):
            previous_df = pd.read_csv(metrics_saving_path)
            for n, row in df_results.iterrows():
                model_name = row['Model']
                model_auc = row['AUC']
                if model_name in previous_df['Model'].tolist():
                    if model_auc >= previous_df[previous_df['Model'] == model_name]['AUC'].item():
                        previous_df = previous_df[previous_df['Model'] != model_name]
                        previous_df = pd.concat([previous_df, pd.DataFrame(row).T])
                        if save_model:
                            with open(models_saving_path / f"best_{model_name}_{labels}.pkl", "wb") as archivo:
                                pickle.dump(best_models[model_name], archivo)
            previous_df.sort_values(by=['Model']).to_csv(metrics_saving_path, index=False)
        else:
            df_results.to_csv(metrics_saving_path, index=False)
            if save_model:
                for model_name, model in best_models.items():
                    with open(models_saving_path / f"best_{model_name}_{labels}.pkl", "wb") as archivo:
                        pickle.dump(model, archivo)
