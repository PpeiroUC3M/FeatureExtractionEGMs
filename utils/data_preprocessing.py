import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import utils.constants as cons


def sliding_window_augmentation(signals, window_size, window_overlap, subsampling, focal, rotor):
    """
    Performs sliding window augmentation on multichannel signal data with optional subsampling and labeling.

    Parameters:
    - signals (ndarray): 2D array of shape (channels, time) representing the multichannel signal.
    - window_size (int): Number of time samples in each window.
    - window_overlap (float): Fraction of overlap between consecutive windows (e.g., 0.5 for 50%).
    - subsampling (int): Subsampling step size (e.g., 2 will take every 2nd sample).
    - focal (array-like): Binary array indicating the presence of focal events over time.
    - rotor (array-like): Binary array indicating the presence of rotor events over time.

    Returns:
    - window_indexes (List[int]): Indexes for each generated window.
    - windowed_signals (List[List[ndarray]]): List of windowed (and subsampled) signal slices.
    - signals_focal_label (List[int]): Binary labels indicating if each window overlaps with a focal event.
    - signals_rotor_label (List[int]): Binary labels indicating if each window overlaps with a rotor event.
    """

    # Calculate the number of samples that will overlap between consecutive windows
    window_overlap_samples = int(window_size * window_overlap)

    # Total number of time steps in the signal
    signals_length = signals.shape[1]

    # Calculate how many windows can fit based on window size and overlap
    num_windows = int((signals_length - window_overlap_samples) / (window_size - window_overlap_samples))

    # Generate window index identifiers
    window_indexes = list(np.arange(num_windows))

    # Initialize containers for sliced windows and labels
    windowed_signals = []
    signals_focal_label = []
    signals_rotor_label = []

    # Loop over each window index
    for w in range(num_windows):
        # Determine the start and end positions of the current window
        window_start = w * (window_size - window_overlap_samples)
        window_end = window_start + window_size

        # Slice the signal for the current window
        windowed_signal = signals[:, window_start:window_end]

        # Apply temporal subsampling
        windowed_signal = windowed_signal[:, ::subsampling]

        # Save the processed window
        windowed_signals.append(list(windowed_signal))

        # Assign a binary label for "focal" if any value in the window is marked 1
        if 1 in focal[window_start:window_end]:
            signals_focal_label.append(1)
        else:
            signals_focal_label.append(0)

        # Same logic for "rotor" label
        if 1 in rotor[window_start:window_end]:
            signals_rotor_label.append(1)
        else:
            signals_rotor_label.append(0)

    return window_indexes, windowed_signals, signals_focal_label, signals_rotor_label


def features_in_windows(binary_activity, discrete_activity, entanglement, lats, window_size, window_overlap,
                        subsampling):
    """
    Extracts statistical and activity-related features from time-series signals using a sliding window approach.

    Parameters:
    - binary_activity (pd.DataFrame): Binary signal representing basic activity (e.g., NLEO detection).
    - discrete_activity (pd.DataFrame): Signal containing 0 (rest), 0.5 (discrete), or 1 (fragmented) states.
    - entanglement (pd.DataFrame): Binary signal indicating presence of entanglement (0 or 1).
    - lats (pd.DataFrame): Activation times with 1s marking significant events (used for dominant cycle estimation).
    - window_size (int): Number of time steps per window.
    - window_overlap (float): Fractional overlap between windows (e.g., 0.5 for 50%).
    - subsampling (int): Downsampling rate for time steps inside windows.

    Returns:
    - windowed_features (pd.DataFrame): Computed features per window including activity metrics and cycle estimations.
    - entanglement_labels (List[int]): Binary label per window indicating presence of entanglement.
    """

    # DataFrame to collect features per window
    windowed_features = pd.DataFrame()

    # Calculate number of overlapping samples per window
    window_overlap_samples = int(window_size * window_overlap)

    # Total number of time steps in the signal
    signals_length = binary_activity.shape[1]

    # Number of sliding windows that can be extracted
    num_windows = int((signals_length - window_overlap_samples) / (window_size - window_overlap_samples))

    # Stores a binary label indicating whether entanglement is present in each window
    entanglement_labels = []

    # Loop over all windows
    for w in range(num_windows):
        # Compute window start and end positions
        window_start = w * (window_size - window_overlap_samples)
        window_end = window_start + window_size

        # Binarize all non-zero entanglement entries to 1
        entanglement[entanglement != 0] = 1

        # Extract and subsample entanglement activity in the window
        windowed_entanglement_activity = entanglement.iloc[:, window_start:window_end].iloc[:, ::subsampling]

        # Compute average entanglement presence across each row (electrode)
        windowed_entanglement_percentage = windowed_entanglement_activity.mean(axis=1)

        # Assign a label: 1 if entanglement is present in any electrode, else 0
        entanglement_label = 1 if windowed_entanglement_percentage.sum() > 0 else 0
        entanglement_labels.append(entanglement_label)

        # Extract activations from 'lats' in the current window
        windowed_lats = lats.iloc[:, window_start:window_end]

        # Get time indices where events occur (where value == 1)
        windowed_lats_indices = [np.where(row == 1)[0] for row in windowed_lats.to_numpy()]

        # Compute dominant cycle intervals by taking mode of differences between events
        aux_bipolar_intervals = np.array([stats.mode(np.diff(indices))[0] for indices in windowed_lats_indices])

        # Replace NaNs (in rows with no events) with the median of existing values
        median_value = np.nanmedian(aux_bipolar_intervals)
        aux_bipolar_intervals[np.isnan(aux_bipolar_intervals)] = median_value
        aux_bipolar_intervals = aux_bipolar_intervals.tolist()

        # Extract and subsample binary activity
        windowed_binary_activity = binary_activity.iloc[:, window_start:window_end].iloc[:, ::subsampling]

        # Compute percentage of active time (mean across time) per electrode
        windowed_binary_activity_percentage = windowed_binary_activity.mean(axis=1)

        # Extract and subsample discrete activity (0 = rest, 0.5 = discrete, 1 = fragmented)
        windowed_discrete = discrete_activity.iloc[:, window_start:window_end].iloc[:, ::subsampling]

        # Compute class-wise percentage per row (rest, discrete, fragmented)
        discrete_fragmented_percentages = windowed_discrete.apply(lambda row: pd.Series({
            'rest': (pd.Series(row == 0)).sum() / len(row) * 100,
            'discrete': (pd.Series(row == 0.5)).sum() / len(row) * 100,
            'fragmented': (pd.Series(row == 1)).sum() / len(row) * 100
        }), axis=1)

        # Extract rest, discrete, and fragmented percentages
        windowed_r = discrete_fragmented_percentages['rest']
        windowed_d = discrete_fragmented_percentages['discrete']
        windowed_f = discrete_fragmented_percentages['fragmented']

        # Compute Activity Index: 1 - |D - F| / (D + F), high = balanced activity
        activity_index = 1 - (np.abs(windowed_d - windowed_f) / (windowed_d + windowed_f))

        # Build new DataFrame for the current window
        new_window = pd.DataFrame({
            'window': w,
            'NLEO': windowed_binary_activity_percentage,
            'rest': windowed_r,
            'discrete': windowed_d,
            'fragmentation': windowed_f,
            'activity_index': activity_index,
            'entanglement_percentage': windowed_entanglement_percentage,
            'dom_cycle': aux_bipolar_intervals
        })

        # Append new features to the main feature DataFrame
        windowed_features = pd.concat([windowed_features, new_window], axis=0)

    return windowed_features, entanglement_labels


def data_augmentation(type_signal, window_size, window_overlap, subsampling, verbose=False, overwrite=False):
    """
    Performs data augmentation by applying sliding window segmentation and extracting features
    from electrogram signals of multiple patients. Saves both augmented signals and computed features.

    Parameters:
    - type_signal (str): 'unipolar' or 'bipolar', determines the type of EGM signals to use.
    - window_size (int): Size of the sliding window (in time steps).
    - window_overlap (float): Fraction of overlap between consecutive windows (e.g., 0.5 for 50%).
    - subsampling (int): Subsampling factor to apply inside each window.
    - verbose (bool): If True, prints progress info.
    - overwrite (bool): If False, skips patients whose data is already saved.
    """

    # Define paths to raw and augmented data
    signals_path = cons.RAW_DATA_PATH
    saving_path = cons.AUGMENTED_DATA_PATH / type_signal

    # Find all patient folders with .pkl signal files
    search_patient_folders_str = os.path.join(signals_path, 'Patient*.pkl')
    patient_folders = glob.glob(search_patient_folders_str)
    num_carto_studies = len(patient_folders)

    # Verbose output: number of studies found
    if verbose:
        print('Num Carto Studies:', num_carto_studies)

    n_acquisitions = []

    # Iterate through each patient study
    for p, pickle_file in enumerate(patient_folders):
        if verbose:
            print(f'- Opening file ({p + 1}/{num_carto_studies}): {pickle_file}')

        # Generate filename for saving output
        aux_str = os.path.basename(pickle_file).split('.')[0]
        aux_save_file_name = (
            f'{aux_str}_augmented_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.parquet'
        )
        aux_save_file_path = os.path.join(saving_path, aux_save_file_name)

        # Path for saving extracted windowed features
        windowed_features_saving_path = cons.AUGMENTED_DATA_PATH / 'features' / f'features_{aux_save_file_name}'

        # Check if file already exists (for skipping unless overwrite)
        aux_file_exists = os.path.isfile(aux_save_file_path)

        # Try loading precomputed features if available
        features = None
        features_path = cons.FEATURES_PATH / (aux_str + '_activity.parquet')
        if features_path.exists():
            features = pd.read_parquet(features_path)

        # If file exists, and we don't want to overwrite it, skip
        if aux_file_exists and not overwrite:
            print(f'- {aux_save_file_path}. File already exists. SKIP!')

        else:
            # Create empty DataFrames to hold augmented signal and feature data
            augmented_data = pd.DataFrame(columns=[
                'patient_id', 'map_name', 'site_id', 'window', type_signal, 'focal', 'rotational'
            ])
            augmented_features = pd.DataFrame(columns=[
                'patient_id', 'map_name', 'site_id', 'window', 'DomCycleL', 'NLEO', 'rest', 'discrete',
                'fragmentation', 'activity_index', 'subsampling'
            ])

            # Load and flatten the pickle content
            aux_data = pd.read_pickle(pickle_file).reset_index()
            aux_num_cf = len(aux_data)


            n_acquisitions.append(aux_num_cf)

    print(n_acquisitions)

            # # Process each recording site (row) in the pickle
            # for cf, cf_data in aux_data.iterrows():
            #     if verbose:
            #         print(f'  - Processing CartoFinder data {cf + 1}/{aux_num_cf}')
            #
            #     # Get the wright type of electrogram signal
            #     if type_signal == 'unipolar':
            #         egm_signals = cf_data.CartoFinderData.Data_EGM
            #     else:
            #         egm_signals = cf_data.BipolarSignals
            #
            #     # Retrieve the annotated events (focal or rotor)
            #     events = cf_data.CartoFinderData.Events
            #     num_events = len(events)
            #
            #     # Initialize zero-filled temporal label arrays
            #     signal_time_length = cf_data.CartoFinderData.EndTime - cf_data.CartoFinderData.StartTime
            #     focal_temporal_label = np.zeros((signal_time_length,))
            #     rotor_temporal_label = np.zeros((signal_time_length,))
            #
            #     # Convert events into binary time series labels
            #     if num_events > 0:
            #         for ee in events:
            #             event_init = ee.EventInit - cf_data.CartoFinderData.StartTime
            #             event_duration = ee.EventEnd - ee.EventInit
            #             if ee.Focal == 1:
            #                 focal_temporal_label[event_init:event_init + event_duration + 1] = 1
            #             if ee.Rotor == 1:
            #                 rotor_temporal_label[event_init:event_init + event_duration + 1] = 1
            #
            #     # Subsample event labels (downsample by 2)
            #     focal_temporal_label = np.append(focal_temporal_label[::2], focal_temporal_label[-1])
            #     rotor_temporal_label = np.append(rotor_temporal_label[::2], rotor_temporal_label[-1])
            #
            #     # Apply sliding window segmentation and labeling
            #     windows_index, windowed_signals, focal, rotational = sliding_window_augmentation(
            #         egm_signals, window_size, window_overlap, subsampling, focal_temporal_label, rotor_temporal_label
            #     )
            #
            #     # Prepare patient/site/window identifiers
            #     num_windows = len(windows_index)
            #     patient_ids = [cf_data.patient_id] * num_windows
            #     maps = [cf_data['map']] * num_windows
            #     site_ids = [cf_data.site_id] * num_windows
            #
            #     # If features exist, extract and compute windowed features
            #     if features is not None:
            #         nleo_activity = features[
            #             (features['Type'] == 'Binary') & (features['map'] == cf_data['map']) &
            #             (features['site_id'] == cf_data.site_id)
            #         ].drop(columns=['Type', 'patient_id', 'map', 'site_id']).dropna(axis=1)
            #
            #         discrete_activity = features[
            #             (features['Type'] == 'Discrete') & (features['map'] == cf_data['map']) &
            #             (features['site_id'] == cf_data.site_id)
            #         ].drop(columns=['Type', 'patient_id', 'map', 'site_id']).dropna(axis=1)
            #
            #         lats = features[
            #             (features['Type'] == 'LATs') & (features['map'] == cf_data['map']) &
            #             (features['site_id'] == cf_data.site_id)
            #         ].drop(columns=['Type', 'patient_id', 'map', 'site_id']).dropna(axis=1)
            #
            #         entanglement = features[
            #             (features['Type'] == 'Entanglement') & (features['map'] == cf_data['map']) &
            #             (features['site_id'] == cf_data.site_id)
            #         ].drop(columns=['Type', 'patient_id', 'map', 'site_id']).dropna(axis=1)
            #
            #         # Extract window-wise engineered features
            #         windowed_features, entanglement_labels = features_in_windows(
            #             nleo_activity, discrete_activity, entanglement, lats,
            #             window_size, window_overlap, subsampling
            #         )
            #
            #         # Build new augmented data rows
            #         new_rows = pd.DataFrame({
            #             'patient_id': patient_ids,
            #             'map_name': maps,
            #             'site_id': site_ids,
            #             'window': windows_index,
            #             type_signal: windowed_signals,
            #             'focal': focal,
            #             'rotational': rotational,
            #             'entanglement': entanglement_labels
            #         })
            #         augmented_data = pd.concat([augmented_data, new_rows], ignore_index=True)
            #
            #         # Annotate features with metadata
            #         windowed_features['DomCycleL'] = cf_data.CartoFinderData.DominantCycleLength
            #         windowed_features['subsampling'] = cf_data.Subsample * subsampling
            #         windowed_features['patient_id'] = cf_data.patient_id
            #         windowed_features['map_name'] = cf_data['map']
            #         windowed_features['site_id'] = cf_data.site_id
            #         augmented_features = pd.concat([augmented_features, windowed_features], ignore_index=True)
            #
            #     else:
            #         # If no features are available, only save signals and labels
            #         new_rows = pd.DataFrame({
            #             'patient_id': patient_ids,
            #             'map_name': maps,
            #             'site_id': site_ids,
            #             'window': windows_index,
            #             type_signal: windowed_signals,
            #             'focal': focal,
            #             'rotational': rotational
            #         })
            #         augmented_data = pd.concat([augmented_data, new_rows], ignore_index=True)
            #
            # # Save the augmented dataset and features to disk as compressed Parquet
            # augmented_data.to_parquet(aux_save_file_path, compression="zstd")
            # if not augmented_features.empty:
            #     augmented_features.to_parquet(windowed_features_saving_path, compression="zstd")


def data_preparation(type_signal, window_size, window_overlap, subsampling, seed, verbose=False):
    """
    Prepares processed datasets (train/val/test) from augmented signal windows by:
    - Loading all pre-augmented .parquet files,
    - Stratifying patients into splits based on total number of signals,
    - Concatenating their data,
    - Saving each split to disk.

    Parameters:
    - type_signal (str): Either 'unipolar' or 'bipolar' (signal type).
    - window_size (int): Window size used during augmentation.
    - window_overlap (float): Fractional overlap between windows.
    - subsampling (int): Subsampling factor used during augmentation.
    - seed (int): Random seed for reproducibility.
    - verbose (bool): Whether to print progress information.
    """

    # Set the random seed for reproducible patient assignment
    np.random.seed(seed)

    # Define input/output paths based on config and type
    signals_path = cons.AUGMENTED_DATA_PATH / type_signal
    saving_path = cons.PROCESSED_DATA_PATH / type_signal

    # Construct the filename suffix based on window parameters
    aux_str = f'_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.parquet'

    # Find all augmented parquet files that match the naming pattern
    patient_folders = np.array(glob.glob(str(signals_path / f'*_augmented{aux_str}')))
    num_carto_studies = len(patient_folders)

    # Verbose output: number of files found
    if verbose:
        print(f"Searching for files: {signals_path / f'*{aux_str}'}")
        print(f"Num Studies: {num_carto_studies}")

    # Count how many windowed signals each patient file contains
    num_signals_per_patient = np.array([len(pd.read_parquet(file)) for file in patient_folders])
    total_signals = num_signals_per_patient.sum()

    if verbose:
        print(f"Num Signals: {total_signals}")

    # -----------------------
    # Helper: Patient Selection
    # -----------------------
    def assign_patients(indices, max_signals):
        """
        Randomly select patients whose total signals do not exceed `max_signals`.
        Returns the selected indices, remaining indices, and the total signal count.
        """
        selected_patients = []
        num_signals = 0
        while num_signals <= max_signals and len(indices) > 0:
            patient_idx = np.random.choice(indices)
            indices = indices[indices != patient_idx]
            selected_patients.append(patient_idx)
            num_signals += num_signals_per_patient[patient_idx]
        return selected_patients, indices, num_signals

    # Create full index of all patients
    remaining_indices = np.arange(num_carto_studies)

    # Assign patients to the test set
    max_test_signals = int(total_signals * cons.TEST_SIZE)
    test_patients, remaining_indices, test_signals = assign_patients(remaining_indices, max_test_signals)

    # Assign patients to the validation set
    max_val_signals = int(total_signals * cons.VAL_SIZE)
    val_patients, remaining_indices, val_signals = assign_patients(remaining_indices, max_val_signals)

    # Remaining patients go to the training set
    train_patients = remaining_indices
    train_signals = num_signals_per_patient[train_patients].sum()

    # Verbose summary of splits
    if verbose:
        print(f"Num test patients: {len(test_patients)}")
        print(f"Num test signals: {test_signals} ({100 * test_signals / total_signals:.2f}%)")
        print(f"Num val patients: {len(val_patients)}")
        print(f"Num val signals: {val_signals} ({100 * val_signals / total_signals:.2f}%)")
        print(f"Num train patients: {len(train_patients)}")
        print(f"Num train signals: {train_signals} ({100 * train_signals / total_signals:.2f}%)")

    # -----------------------
    # Helper: Load and Concatenate
    # -----------------------
    def load_and_concatenate_data(file_paths):
        """
        Concatenates all Parquet files in the list into one DataFrame.
        """
        signals = pd.DataFrame()
        for file_path in file_paths:
            signals = pd.concat([signals, pd.read_parquet(file_path)], ignore_index=True)
        return signals

    # Load data for each split
    x_train = load_and_concatenate_data(patient_folders[train_patients])
    x_val = load_and_concatenate_data(patient_folders[val_patients])
    x_test = load_and_concatenate_data(patient_folders[test_patients])

    # -----------------------
    # Helper: Save Data
    # -----------------------
    def save_data(data, file_name, save_path):
        """
        Saves a DataFrame to a compressed Parquet file.
        """
        data.to_parquet(save_path / file_name, compression="zstd")
        if verbose:
            print(f"{file_name.split('_')[0].upper()} SAVED!")

    # Save each split to disk
    save_data(x_test, f'Test_seed{seed}' + aux_str, saving_path)
    save_data(x_val, f'Val_seed{seed}' + aux_str, saving_path)
    save_data(x_train, f'Train_seed{seed}' + aux_str, saving_path)


def normalization(x_train, x_val, x_test, clipping=True, verbose=False, save_figure=True, saving_path=None):
    """
    Function to normalize the signals in the train, validation and test subsets.

    ARGUMENTS
    ---------
    :param x_train: np.array, dimensions (num_signals x num_channels x window_size x 1)
        Train subset of the signals
    :param x_val: np.array, dimensions (num_signals x num_channels x window_size x 1)
        Validation subset of the signals
    :param x_test: np.array, dimensions (num_signals x num_channels x window_size x 1)
        Test subset of the signals
    :param clipping: bool
         If clip the signals to the percentiles
    :param verbose: bool
        If process is logged
    :param save_figure: bool
        If save the histogram and percentiles figure
    :param saving_path:
        Path to save the histogram and percentiles figure

    RETURNS
    -------
    :return: np.array, dimensions (num_signals x num_channels x window_size x 1)
        Normalized signals in the train, validation and test subsets
    """
    # Calculate percentiles on the train subset
    perc = np.percentile(x_train.flatten(), [2, 98])
    if verbose:
        print(f'2% Percentile ({perc[0]:.2f})')
        print(f'98% Percentile ({perc[1]:.2f})')

    # Filtra valores dentro del 2% y 98%
    filtered = x_train.flatten()[(x_train.flatten() >= perc[0]) & (x_train.flatten() <= perc[1])]

    plt.figure(figsize=(8, 6))
    plt.hist(filtered, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Percentiles (seguros porque estamos en el mismo rango)
    plt.axvline(perc[0].item(), color='red', linestyle='dashed', linewidth=1.5,
                label=f'2% Percentile ({perc[0]:.2f})')
    plt.axvline(perc[1].item(), color='green', linestyle='dashed', linewidth=1.5,
                label=f'98% Percentile ({perc[1]:.2f})')

    plt.title('Histogram with Percentiles (Filtered)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    if save_figure:
        saving_path = saving_path.split('.')[0] + '.png'
        plt.savefig(saving_path, dpi=300)
    plt.show()

    # Clip signals
    if clipping:
        x_train[x_train < perc[0]] = perc[0]
        x_train[x_train > perc[1]] = perc[1]
        x_val[x_val < perc[0]] = perc[0]
        x_val[x_val > perc[1]] = perc[1]
        x_test[x_test < perc[0]] = perc[0]
        x_test[x_test > perc[1]] = perc[1]

    # Normalize in the range [-1, 1] by min-max scaler
    x_train_norm = (x_train - perc[0]) / (perc[1] - perc[0]) * 2 - 1
    x_val_norm = (x_val - perc[0]) / (perc[1] - perc[0]) * 2 - 1
    x_test_norm = (x_test - perc[0]) / (perc[1] - perc[0]) * 2 - 1

    return x_train_norm, x_val_norm, x_test_norm


def normalize_data(type_signal, window_size, window_overlap, subsampling, seed, verbose=False, save_figures=True):
    """
    Loads pre-split (train/val/test) windowed signal datasets, applies normalization,
    prepares target labels, and saves everything as pickled files.

    Parameters:
    - type_signal (str): Either 'unipolar' or 'bipolar'.
    - window_size (int): Size of the time window used during signal augmentation.
    - window_overlap (float): Overlap ratio between windows.
    - subsampling (int): Temporal subsampling factor used in the windowing.
    - seed (int): Seed used for dataset splitting (for consistent filenames).
    - verbose (bool): If True, prints data stats and progress.
    - save_figures (bool): If True, saves histogram plot of normalization.
    """

    assert type_signal in ['unipolar', 'bipolar'], 'Invalid type of signal'

    signals_path = cons.PROCESSED_DATA_PATH / type_signal
    saving_path = signals_path

    # ---------------------------
    # Helper: Load parquet data
    # ---------------------------
    def load_data(loading_path, file_name):
        file_path = os.path.join(loading_path, file_name)
        return pd.read_parquet(str(file_path))

    aux_str = f'_seed{seed}_L{int(window_size/subsampling)}_O{window_overlap}_S{2*subsampling}.parquet'

    # Load split datasets
    val_data = load_data(signals_path, 'Val' + aux_str)
    test_data = load_data(signals_path, 'Test' + aux_str)
    train_data = load_data(signals_path, 'Train' + aux_str)

    # Convert signal lists to 4D numpy arrays [samples, channels, time, 1]
    x_val = np.stack(val_data[type_signal].apply(lambda x: np.vstack(x)))[:, :, :, None]
    x_test = np.stack(test_data[type_signal].apply(lambda x: np.vstack(x)))[:, :, :, None]
    x_train = np.stack(train_data[type_signal].apply(lambda x: np.vstack(x)))[:, :, :, None]

    if verbose:
        print('Train dimensions:', x_train.shape)
        print('Validation dimensions:', x_val.shape)
        print('Test dimensions:', x_test.shape)

    # Apply normalization using training distribution
    histogram_file = os.path.join(saving_path, 'histogram' + aux_str)
    x_train_norm, x_val_norm, x_test_norm = normalization(
        x_train, x_val, x_test, True, verbose, save_figures, histogram_file
    )

    # Extract binary labels and reshape them to (N, 1)
    y_train_focal = np.array(train_data.focal)[:, None]
    y_train_rotor = np.array(train_data.rotational)[:, None]
    y_train_entanglement = np.array(train_data.entanglement)[:, None]

    y_val_focal = np.array(val_data.focal)[:, None]
    y_val_rotor = np.array(val_data.rotational)[:, None]
    y_val_entanglement = np.array(val_data.entanglement)[:, None]

    y_test_focal = np.array(test_data.focal)[:, None]
    y_test_rotor = np.array(test_data.rotational)[:, None]
    y_test_entanglement = np.array(test_data.entanglement)[:, None]

    # ---------------------------
    # Helper: Save pickled data
    # ---------------------------
    def save_data(data, file_name, save_path):
        with open(save_path / file_name, 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            print(f"{file_name.split('_')[0].upper()} NORM SAVED!")

    # Save all datasets and labels
    aux_str = f'_seed{seed}_L{int(window_size / subsampling)}_O{window_overlap}_S{2 * subsampling}.pkl'

    save_data(x_train_norm, 'x_train' + aux_str, saving_path)
    save_data(x_val_norm, 'x_val' + aux_str, saving_path)
    save_data(x_test_norm, 'x_test' + aux_str, saving_path)

    save_data(y_train_focal, 'y_train_focal' + aux_str, saving_path)
    save_data(y_val_focal, 'y_val_focal' + aux_str, saving_path)
    save_data(y_test_focal, 'y_test_focal' + aux_str, saving_path)

    save_data(y_train_rotor, 'y_train_rotor' + aux_str, saving_path)
    save_data(y_val_rotor, 'y_val_rotor' + aux_str, saving_path)
    save_data(y_test_rotor, 'y_test_rotor' + aux_str, saving_path)

    save_data(y_train_entanglement, 'y_train_entanglement' + aux_str, saving_path)
    save_data(y_val_entanglement, 'y_val_entanglement' + aux_str, saving_path)
    save_data(y_test_entanglement, 'y_test_entanglement' + aux_str, saving_path)
