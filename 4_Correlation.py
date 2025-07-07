import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils.constants as cons
from utils.load_data import load_data, load_embeddings


vae = False

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
batch_size = 64

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

# Iterate over all architecture combinations
for layer_channel in layers_channels:
    for latent_dim in latent_dims:

        # Build auxiliary name strings for file paths
        aux_str_data = f'_L{int(window_size / subsampling)}_O{window_overlap}_S{2 * subsampling}.parquet'
        aux_str_model = (f'ch{layer_channel}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}'
                         f'_ld{latent_dim}')

        # Load validation and test data and embeddings
        _, val_data, test_data = load_data(type_signal, seed, window_size, window_overlap, subsampling, ['val', 'test'])
        _, val_embeds, test_embeds = load_embeddings(type_signal, layer_channel, kernel_size, s, padding, pooling,
                                                     norm, drop, latent_dim, ['val', 'test'], vae)
        val_embeds, test_embeds = pd.DataFrame(val_embeds), pd.DataFrame(test_embeds)
        all_embeds = pd.concat([test_embeds, val_embeds]).reset_index(drop=True)

        # Load and merge feature files for all patients in val and test sets
        df_features = pd.DataFrame()
        for patient in test_data.patient_id.unique().tolist() + val_data.patient_id.unique().tolist():
            features_path = (cons.AUGMENTED_DATA_PATH / 'features' /
                             f'features_{patient}_CartoFinder_augmented{aux_str_data}')
            if features_path.exists():
                patient_features = pd.read_parquet(features_path).drop(columns=['subsampling', 'DomCycleL']).fillna(0)
                df_features = pd.concat([df_features, patient_features])
            else:
                print(f'Features file {features_path} not found.')

        # Create folder to save correlation heatmaps
        if vae:
            figures_saving_path = cons.RESULTS_DIR / 'vae' / type_signal / aux_str_model / 'correlations'
        else:
            figures_saving_path = cons.RESULTS_DIR / type_signal / aux_str_model / 'correlations'
        figures_saving_path.mkdir(parents=True, exist_ok=True)

        # Prepare figure for feature summary correlation heatmaps
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=(20, 0.3), wspace=0.05)

        # Define feature aggregation types
        types_features = ['mean']

        # Compute and plot correlations between embeddings and summary feature statistics
        for i, type_features in enumerate(types_features):
            if type_features == 'mean':
                patient_features = df_features.groupby(['patient_id', 'map_name', 'site_id', 'window'],
                                                       sort=False).mean()
            else:
                patient_features = None

            # Compute correlation matrix between embeddings and features
            patient_features = patient_features.reset_index(drop=True)
            correlation_matrix = pd.DataFrame(index=all_embeds.columns, columns=patient_features.columns)

            for col1 in all_embeds.columns:
                for col2 in patient_features.columns:
                    correlation_matrix.loc[col1, col2] = all_embeds[col1].corr(patient_features[col2])

            correlation_matrix = correlation_matrix.astype(float)  # Ensure numeric for plotting

            # Plot heatmap
            ax = fig.add_subplot(gs[0])
            sns.heatmap(correlation_matrix.T,
                        vmin=-1, vmax=1,
                        cmap='coolwarm',
                        annot=True, fmt=".2f",
                        linewidths=0.5,
                        linecolor='gray',
                        square=True,
                        cbar=False,  # Desactivamos la colorbar por defecto
                        ax=ax)

            ax_cb = fig.add_subplot(gs[1])
            norm = plt.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=ax_cb, label='Correlation')

        plt.tight_layout()
        plt.savefig(figures_saving_path / f'correlations_features_stats.png')
        plt.close()

        # Expand grouped list features into separate columns
        df_features = df_features.groupby(['patient_id', 'map_name', 'site_id', 'window'],
                                          sort=False).agg(list).reset_index(drop=True)
        expanded = pd.DataFrame()
        for col in df_features.columns:
            max_len = df_features[col].str.len().max()
            new_cols = pd.DataFrame(df_features[col].tolist(), columns=[f'{col}_{i + 1}' for i in range(max_len)])
            expanded = pd.concat([expanded, new_cols], axis=1)
        expanded.reset_index(drop=True)

        # Correlation heatmap for expanded features
        correlation_matrix = pd.DataFrame(index=all_embeds.columns, columns=expanded.columns)
        for col1 in all_embeds.columns:
            for col2 in expanded.columns:
                correlation_matrix.loc[col1, col2] = all_embeds[col1].corr(expanded[col2])

        correlation_matrix = correlation_matrix.astype(float)

        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title(f'Correlation between dimensions and features')
        plt.xlabel('features')
        plt.ylabel('dimensions')
        plt.tight_layout()
        plt.savefig(figures_saving_path / f'correlations_features.png')
        plt.close()
