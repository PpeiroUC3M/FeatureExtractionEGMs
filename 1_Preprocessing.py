from utils.data_preprocessing import data_augmentation, data_preparation, normalize_data
from utils.features_extraction import features_extraction

# Parameters for preprocessing
type_signal = 'unipolar'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
verbose = True

# Decide what to do
FEATURES = False
AUG = True
PREPARE = False
NORMALIZE = False

# Extract features
if FEATURES:
    features_extraction(500, 1, verbose)

# Preprocess data
if AUG:
    data_augmentation(type_signal, window_size, window_overlap, subsampling, verbose, overwrite=True)
if PREPARE:
    data_preparation(type_signal, window_size, window_overlap, subsampling, seed, verbose)
if NORMALIZE:
    normalize_data(type_signal, window_size, window_overlap, subsampling, seed, verbose)
