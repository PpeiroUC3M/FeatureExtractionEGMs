import os
import torch
import pickle
import numpy as np
import pandas as pd
import utils.constants as cons
from utils.autoencoders import ConvAutoencoder

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


def load_pickle(path):
    """
    Loads and returns a Python object from a pickle (.pkl) file.

    Parameters:
    - path (str or Path): Path to the pickle file.

    Returns:
    - Loaded Python object from the file.
    """
    # Open the pickle file in binary read mode and load its contents
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_data(type_signal, seed, window_size, window_overlap, subsampling, subsets=None):
    """
    Loads preprocessed signal data (train, validation, test) from Parquet files
    based on signal type and preprocessing configuration.

    Parameters:
    - type_signal (str): 'unipolar' or 'bipolar'.
    - seed (int): Seed used during dataset preparation (included in filenames).
    - window_size (int): Size of the sliding window used in preprocessing.
    - window_overlap (float): Fractional overlap between windows.
    - subsampling (int): Temporal subsampling factor.
    - subsets (list): Subsets to load.

    Returns:
    - train_data (DataFrame): Training set loaded from Parquet.
    - val_data (DataFrame): Validation set loaded from Parquet.
    - test_data (DataFrame): Test set loaded from Parquet.
    """
    train_data, val_data, test_data = None, None, None

    # Construct the filename suffix from the preprocessing parameters
    aux_str_data = f'_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.parquet'
    data_path = cons.PROCESSED_DATA_PATH / type_signal

    # Load the train, val, and test DataFrames from disk
    if 'train' in subsets:
        train_data = pd.read_parquet(data_path / (f'Train_seed{seed}' + aux_str_data))
    if 'val' in subsets:
        val_data = pd.read_parquet(data_path / (f'Val_seed{seed}' + aux_str_data))
    if 'test' in subsets:
        test_data = pd.read_parquet(data_path / (f'Test_seed{seed}' + aux_str_data))

    return train_data, val_data, test_data


def load_signals(type_signal, seed, window_size, window_overlap, subsampling, subsets=None):
    """
    Loads preprocessed signal tensors (train, validation, test) from pickle files.

    Parameters:
    - type_signal (str): Either 'unipolar' or 'bipolar'.
    - seed (int): Seed used during data splitting.
    - window_size (int): Size of each sliding window used in preprocessing.
    - window_overlap (float): Overlap ratio between windows (e.g., 0.5 for 50%).
    - subsampling (int): Subsampling factor used during signal slicing.
    - subsets (list): Subsets to load.

    Returns:
    - x_train, x_val, x_test: 4D numpy arrays containing signals for each split.
    """
    x_train, x_val, x_test = None, None, None

    # Generate filename suffix based on preprocessing parameters
    aux_str_data = f'_seed{seed}_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.pkl'
    data_path = cons.PROCESSED_DATA_PATH / type_signal

    # Load the processed signals for each split
    if 'train' in subsets:
        x_train = load_pickle(data_path / ('x_train' + aux_str_data))
    if 'val' in subsets:
        x_val = load_pickle(data_path / ('x_val' + aux_str_data))
    if 'test' in subsets:
        x_test = load_pickle(data_path / ('x_test' + aux_str_data))

    return x_train, x_val, x_test


def load_labels(type_signal, labels, seed, window_size, window_overlap, subsampling):
    """
    Loads label arrays (train, validation, test) from pickled files based on a specific label type.

    Parameters:
    - type_signal (str): Either 'unipolar' or 'bipolar'.
    - labels (str): Label name to load ('focal', 'rotor', or 'entanglement').
    - seed (int): Seed used during data preparation.
    - window_size (int): Size of each time window.
    - window_overlap (float): Overlap ratio between windows.
    - subsampling (int): Subsampling factor used during preprocessing.

    Returns:
    - y_train, y_val, y_test: 1D numpy arrays of labels for each dataset split.
    """

    # Construct file name suffix based on preprocessing config
    aux_str_data = f'_seed{seed}_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.pkl'
    data_path = cons.PROCESSED_DATA_PATH / type_signal

    # Load and flatten label arrays from pickle files
    y_train = load_pickle(data_path / (f'y_train_{labels}' + aux_str_data)).ravel()
    y_val = load_pickle(data_path / (f'y_val_{labels}' + aux_str_data)).ravel()
    y_test = load_pickle(data_path / (f'y_test_{labels}' + aux_str_data)).ravel()

    return y_train, y_val, y_test


def load_embeddings(type_signal, layer_channels, kernel_size, s, padding, pooling, norm, drop, latent_dim,
                    subsets=None, vae=False):
    """
    Loads train, validation, and test embeddings from disk based on the autoencoder configuration.

    Parameters:
    - type_signal (str): 'unipolar' or 'bipolar'.
    - layer_channels (tuple): Channels used in the encoder/decoder layers.
    - kernel_size (int): Kernel size used in the autoencoder.
    - s (int): Stride used in the autoencoder.
    - padding (int): Padding used in the autoencoder.
    - pooling (tuple): Pooling configuration per layer.
    - norm (bool): Whether normalization was used.
    - drop (float): Dropout rate used.
    - latent_dim (int): Size of the latent space.
    - subsets (list): Subsets to load.

    Returns:
    - train_embeds, val_embeds, test_embeds: Loaded embeddings for each dataset split.
    """
    train_embeds, val_embeds, test_embeds = None, None, None

    # Build directory name based on autoencoder configuration
    aux_str_ae = f'ch{layer_channels}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}_ld{latent_dim}'
    if vae:
        embeddings_path = cons.RESULTS_DIR / 'vae' / type_signal / aux_str_ae / 'embeddings'
    else:
        embeddings_path = cons.RESULTS_DIR / type_signal / aux_str_ae / 'embeddings'

    # Load the embeddings for each split from .pkl files
    if 'train' in subsets:
        train_embeds = load_pickle(embeddings_path / 'train_embeds.pkl')
    if 'val' in subsets:
        val_embeds = load_pickle(embeddings_path / 'val_embeds.pkl')
    if 'test' in subsets:
        test_embeds = load_pickle(embeddings_path / 'test_embeds.pkl')

    return train_embeds, val_embeds, test_embeds


def load_models(type_signal, labels, model, layer_channels, kernel_size, s, padding, pooling, norm, drop, latent_dim):
    """
    Loads a saved model object from disk based on the autoencoder and classifier configuration.

    Parameters:
    - type_signal (str): Either 'unipolar' or 'bipolar'.
    - labels (str): Label type the model was trained to predict (e.g. 'focal', 'rotor').
    - model (str): Classifier name (e.g. 'logreg', 'svm', etc.).
    - layer_channels (tuple): Autoencoder channel configuration.
    - kernel_size (int): Convolutional kernel size.
    - s (int): Stride used in the convolution layers.
    - padding (int): Padding used in the convolution layers.
    - pooling (tuple): Pooling configuration used in the encoder.
    - norm (bool): Whether normalization was applied in the encoder.
    - drop (float): Dropout probability.
    - latent_dim (int): Latent dimensionality of the autoencoder.

    Returns:
    - model: The loaded classifier model.
    """

    # Build a directory path string based on the encoder configuration
    aux_str_ae = f'ch{layer_channels}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}_ld{latent_dim}'
    models_path = cons.RESULTS_DIR / type_signal / aux_str_ae / 'model'

    # Load the pickled classifier model
    model = load_pickle(models_path / f'best_{model}_{labels}.pkl')

    return model


def load_autoencoder(type_signal, window_size, subsampling, layer_channels, kernel_size, s, padding, pooling,
                     norm, drop, latent_dim, train=False):
    """
    Initializes a ConvAutoencoder and loads its pre-trained weights from disk if available.

    Parameters:
    - type_signal (str): 'unipolar' or 'bipolar', determines input shape.
    - window_size (int): Temporal length of the input window.
    - subsampling (int): Temporal subsampling applied to input.
    - layer_channels (tuple): Channel configuration per convolutional layer.
    - kernel_size (int): Kernel size for convolutions.
    - s (int): Stride for convolutions.
    - padding (int): Padding applied to convolutions.
    - pooling (tuple): Pooling configuration per encoder layer.
    - norm (bool): Whether batch normalization is used.
    - drop (float): Dropout rate.
    - latent_dim (int): Size of the latent vector.

    Returns:
    - model (ConvAutoencoder): Loaded autoencoder model, set to eval mode if pre-trained weights are found.
    """

    input_channels = 1

    # Set input shape depending on signal type (15 channels for bipolar, 20 for unipolar)
    if type_signal == 'bipolar':
        input_shape = (15, int(window_size / subsampling))
    else:
        input_shape = (20, int(window_size / subsampling))

    # Instantiate the ConvAutoencoder with specified architecture
    model = ConvAutoencoder(
        in_channels=input_channels,
        input_size=input_shape,
        channels_per_layer=layer_channels,
        kernel_size=kernel_size,
        stride=s,
        padding=padding,
        norm=norm,
        pooling=pooling,
        drop=drop,
        latent_dim=latent_dim
    )

    # Construct the model saving path based on architecture parameters
    aux_str = (f'ch{layer_channels}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}'
               f'_ld{latent_dim}')
    model_saving_path = cons.RESULTS_DIR / type_signal / aux_str / 'model' / "best_autoencoder.pth"

    # If train
    if train:
        model.train()
    else:
        # Load saved weights if available
        if os.path.exists(model_saving_path):
            model.load_state_dict(torch.load(model_saving_path))
        model.eval()  # Switch to evaluation mode after loading

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


class DatasetEGM(Dataset):
    """
    Custom PyTorch Dataset for loading EGM signal data and optional labels from pickle files.

    Parameters:
    - data_path (str or Path): Path to the pickled file containing the signal data.
    - num_signals (int or None): If provided, limits the dataset to the first N signals.
    - labels_path (str or Path, optional): Path to the pickled file containing the labels (if available).
    """

    def __init__(self, data_path, num_signals, labels_path=None):
        # Load signal data from pickle file
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            if num_signals is not None:
                self.data = self.data[:num_signals]

        self.labels_path = labels_path

        # If labels are provided, load them and apply the same slicing
        if labels_path is not None:
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
                if num_signals is not None:
                    self.labels = self.labels[:num_signals]

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get a single sample and convert it to a tensor [1, channels, time]
        image = self.data[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32).squeeze().unsqueeze(0)

        # Return the sample with or without its corresponding label
        if self.labels_path is not None:
            label = self.labels[idx]
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return image_tensor


def load_dataloaders(type_signal, window_size, window_overlap, subsampling, seed, batch_size, shuffle, num_signals=None,
                     labels=None):
    """
    Loads PyTorch DataLoaders for training, validation, and test sets of EGM signals (and optional labels).

    Parameters:
    - type_signal (str): 'unipolar' or 'bipolar'.
    - window_size (int): Temporal window size used in preprocessing.
    - window_overlap (float): Overlap between consecutive windows.
    - subsampling (int): Subsampling factor used when slicing signals.
    - seed (int): Seed used for data split (used in file naming).
    - batch_size (int): Batch size for DataLoader.
    - shuffle (bool): Whether to shuffle the data within each loader.
    - num_signals (int or None): Limit the number of signals per split.
    - labels (str or None): One of ['focal', 'rotor', 'entanglement'] if labels should be loaded.

    Returns:
    - train_dataloader, val_dataloader, test_dataloader: PyTorch DataLoader objects.
    """

    # Construct common file suffix based on preprocessing parameters
    aux_str = f'_seed{seed}_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.pkl'
    base_path = cons.PROCESSED_DATA_PATH / type_signal

    # ---------------------------
    # Helper: Build DataLoader for a given split
    # ---------------------------
    def build_loader(split):
        data_path = base_path / f'x_{split}{aux_str}'
        label_path = base_path / f'y_{split}_{labels}{aux_str}' if labels in ['focal', 'rotor', 'entanglement'] \
            else None
        dataset = DatasetEGM(data_path, num_signals, label_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Create DataLoaders for each dataset split
    train_dataloader = build_loader('train')
    val_dataloader = build_loader('val')
    test_dataloader = build_loader('test')

    return train_dataloader, val_dataloader, test_dataloader


def load_scaler(type_signal, layer_channels, kernel_size, s, padding, pooling, norm, drop, latent_dim):
    aux_str_model = (f'ch{layer_channels}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}'
                     f'_ld{latent_dim}')
    embeddings_saving_path = cons.RESULTS_DIR / type_signal / aux_str_model / 'embeddings'

    if os.path.exists(embeddings_saving_path / 'scaler.pkl'):
        with open(embeddings_saving_path / 'scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    else:
        train_embeds, val_embeds, test_embeds = load_embeddings(type_signal, layer_channels, kernel_size, s, padding,
                                                                pooling, norm, drop, latent_dim,
                                                                subsets=['train', 'val', 'test'])
        all_embeds = np.concatenate([train_embeds, val_embeds, test_embeds], axis=0)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(all_embeds)
        with open(embeddings_saving_path / 'scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

    return scaler
