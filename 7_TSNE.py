import pandas as pd
import utils.constants as cons
from utils.load_data import load_labels, load_embeddings
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler
import umap.umap_ as umap

import plotly.express as px

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'unipolar'
labels = ['focal']
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
verbose = True

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

# Loop over all AE configurations and label types
for layer_channel in layers_channels:
    for latent_dim in latent_dims:
        for label in labels:

            # Load corresponding labels (train/val/test split)
            _, _, y_test = load_labels(type_signal, label, seed, window_size, window_overlap, subsampling)

            # Load embeddings from test set
            _, _, test_embeds = load_embeddings(type_signal, layer_channel, kernel_size, s, padding, pooling, norm,
                                                drop, latent_dim, ['test'])

            # Build directory name based on AE config
            aux_str_model = (
                f'ch{layer_channel}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}_ld{latent_dim}'
            )
            saving_path = cons.RESULTS_DIR / type_signal / aux_str_model / 'tsne'
            saving_path.mkdir(parents=True, exist_ok=True)

            # ---------------------------
            # Helper: Plot and save t-SNE visualization
            # ---------------------------
            def plot_tsne_3d(x_data, y_data, num_points=1000):
                # Balance dataset using undersampling to have equal classes
                rus = RandomUnderSampler(
                    sampling_strategy={0: num_points // 2, 1: num_points // 2},
                    random_state=0
                )
                x_resampled, y_resampled = rus.fit_resample(x_data, y_data)

                # Fit t-SNE on balanced embeddings
                tsne = TSNE(n_components=3, random_state=seed)
                x_tsne = tsne.fit_transform(x_resampled)

                # Create DataFrame for plotting
                df = pd.DataFrame(x_tsne, columns=['X', 'Y', 'Z'])
                df['Label'] = y_resampled.flatten().astype(str)

                # 3D scatter plot with Plotly
                fig = px.scatter_3d(
                    df, x='X', y='Y', z='Z', color='Label',
                    title=f"T-SNE representation: {label} activity",
                    opacity=0.5
                )

                # Save HTML file of interactive plot
                fig.write_html(saving_path / f'tsne_{label}.html')


            def plot_tsne_2d(x_data, y_data, num_points=500):
                # Balance dataset using undersampling to have equal classes
                rus = RandomUnderSampler(
                    sampling_strategy={0: num_points // 2, 1: num_points // 2},
                    random_state=42
                )
                x_resampled, y_resampled = rus.fit_resample(x_data, y_data)

                # Fit t-SNE on balanced embeddings (2D)
                tsne = TSNE(n_components=2, random_state=42)
                x_tsne = tsne.fit_transform(x_resampled)

                # Create the DataFrame
                df = pd.DataFrame(x_tsne, columns=['X', 'Y'])
                df['Label'] = y_resampled.flatten()

                # Map numeric labels to descriptive strings
                label_mapping = {0: 'Non-focal', 1: 'Focal'}
                df['Label'] = df['Label'].map(label_mapping)

                # Define desired label order and colors
                ordered_labels = ['Non-focal', 'Focal']
                color_mapping = {'Non-focal': '#1f77b4', 'Focal': '#ff7f0e'}

                # Plot using matplotlib
                plt.figure(figsize=(8, 6))
                for label_val in ordered_labels:
                    subset = df[df['Label'] == label_val]
                    plt.scatter(subset['X'], subset['Y'],
                                label=label_val,
                                color=color_mapping[label_val],
                                alpha=0.6)

                plt.xlabel("t-SNE Component 1", fontsize=16)
                plt.ylabel("t-SNE Component 2", fontsize=16)
                plt.legend(loc='upper right', fontsize=14)

                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)

                # Save PNG
                plt.savefig(saving_path / f'tsne_{label}.png', dpi=300)
                plt.close()

            # Run t-SNE visualization
            plot_tsne_2d(test_embeds, y_test, 1000)
