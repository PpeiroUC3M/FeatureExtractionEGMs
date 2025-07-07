import os
import glob
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from utils.load_data import load_embeddings, load_autoencoder, load_models, load_scaler
import utils.constants as cons
from sklearn.preprocessing import StandardScaler

GRAPHS = True

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
labels = 'entanglement'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42

if type_signal == 'unipolar':
    upper_clip = 0.16
    lower_clip = -0.15
else:
    upper_clip = 0.87
    lower_clip = -0.95

# ---------------------------
# AE parameters
# ---------------------------
layer_channels = [64, 64]
latent_dim = 16
kernel_size = 3
s = 1
padding = 1
pooling = True
norm = True
drop = 0.25

# ---------------------------
# Classification model
# ---------------------------
model = 'CatBoost'


def windowing(signals, window_size, subsampling):
    signals_length = signals.shape[1]
    num_windows = signals_length // window_size

    windowed_signals = signals[:, :num_windows * window_size].reshape(signals.shape[0], num_windows, window_size)
    windowed_signals = windowed_signals[:, :, ::subsampling]

    return np.transpose(windowed_signals, (1, 0, 2)), num_windows


def normalization(signals):

    signals[signals < lower_clip] = lower_clip
    signals[signals > upper_clip] = upper_clip
    signals_norm = (signals - lower_clip) / (upper_clip - lower_clip) * 2 - 1

    return signals_norm


# Load autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = load_autoencoder(type_signal, window_size, subsampling, layer_channels, kernel_size, s, padding, pooling,
                      norm, drop, latent_dim)

# Create scaler
train_embeds, _, _ = load_embeddings(type_signal, layer_channels, kernel_size, s, padding, pooling, norm, drop,
                                     latent_dim, ['train'])
scaler = StandardScaler()
scaler.fit(train_embeds)

# Load classifier
classifier = load_models(type_signal, labels, model, layer_channels, kernel_size, s, padding, pooling, norm, drop,
                         latent_dim)

# See patients folders
signals_path = cons.RAW_DATA_PATH
search_patient_folders_str = os.path.join(signals_path, 'Patient*.pkl')
patient_folders = glob.glob(search_patient_folders_str)

times = []
total_windows = 0
for n_patient in range(len(patient_folders)):
    pickle_file = patient_folders[n_patient]
    aux_data = pd.read_pickle(pickle_file).reset_index()

    for cf, cf_data in aux_data.iterrows():

        if type_signal == 'unipolar':
            egm_signals = cf_data.CartoFinderData.Data_EGM
        else:
            egm_signals = cf_data.BipolarSignals

        if egm_signals.shape[1] == 15001:
            start = time.time()
            windowed_signals, num_windows = windowing(egm_signals, window_size, subsampling)
            windowed_norm_signals = normalization(windowed_signals)

            torch_signals = torch.from_numpy(windowed_norm_signals).unsqueeze(1).float().to(device)

            embeds, _, _ = ae(torch_signals)
            numpy_embeds = embeds.to('cpu').detach().numpy()
            embeds_norm = scaler.transform(numpy_embeds)

            outputs = classifier.predict(embeds_norm)
            end = time.time()

            classification_time = end - start
            times.append(classification_time)
            total_windows += num_windows

            if GRAPHS:

                def detectar_primer_patron(outputs):
                    outputs_int = outputs.astype(int)
                    pattern_str = ''.join(map(str, outputs_int))
                    patrones = ['111000', '000111']

                    for i in range(len(pattern_str) - 5):
                        subcadena = pattern_str[i:i + 6]
                        if subcadena in patrones:
                            inicio = i
                            corte = i + 3
                            final = i + 6
                            return True, (inicio, corte, final)

                    return False, None


                if detectar_primer_patron(outputs)[0]:
                    inicio, corte, final = detectar_primer_patron(outputs)[1]

                    sampling_rate = 250  # Hz
                    offset = 2

                    def plot_window_range(window_range, title):
                        plt.figure(figsize=(20, 14))

                        # Compute actual start and end time in seconds
                        start_sample = window_range.start * window_size // subsampling
                        end_sample = window_range.stop * window_size // subsampling
                        start_time = start_sample / sampling_rate
                        end_time = end_sample / sampling_rate

                        for i in window_range:
                            start = i * window_size // subsampling
                            end = start + window_size // subsampling
                            t = np.arange(start, end) / sampling_rate  # absolute time

                            if outputs[i] == 1:
                                plt.axvspan(t[0], t[-1], color='red', alpha=0.2)

                            for ch in range(windowed_norm_signals.shape[1]):
                                plt.plot(t, windowed_norm_signals[i, ch] - ch * offset, 'b')

                        # Red dashed vertical lines every 1s
                        for x in np.arange(np.floor(start_time), np.ceil(end_time) + 1):
                            plt.axvline(x=x, color='red', linewidth=2)

                        plt.xlim(start_time, end_time)

                        # Font and axis settings
                        plt.xlabel("Time (s)", fontsize=22)
                        plt.ylabel("Channels", fontsize=22)
                        plt.title(title, fontsize=26)
                        plt.xticks(fontsize=20)
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f'{title}.png')
                        plt.show()

                    # Plot all windows
                    plot_window_range(range(windowed_norm_signals.shape[0]), "Full signal")

                    # Plot first 5 windows
                    plot_window_range(range(inicio, corte), "No entanglement")

                    # Plot windows 5 to 7
                    plot_window_range(range(corte, final), f"Entanglement")


print(np.array(times).mean())
