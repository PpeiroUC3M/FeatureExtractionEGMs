import os
import glob
import numpy as np
import pandas as pd
import utils.constants as cons

from functions.RAISEntanglement import detectEntanglementRAISV2


def features_extraction(fs=1000, sampling=1, verbose=False, overwrite=False):
    """
    Extracts electrophysiological features from unipolar and bipolar signals using
    entanglement and activity analysis, then saves them to parquet files.

    Parameters:
    - fs (int): Sampling frequency of the signal (default is 1000 Hz).
    - sampling (int): Additional downsampling factor applied during preprocessing.
    - verbose (bool): Whether to print processing progress.
    - overwrite (bool): Whether to overwrite existing saved files.
    """

    signals_path = cons.RAW_DATA_PATH
    saving_path = cons.FEATURES_PATH

    # Locate all patient signal files
    search_patient_folders_str = os.path.join(signals_path, 'Patient*.pkl')
    patient_folders = glob.glob(search_patient_folders_str)
    num_carto_studies = len(patient_folders)

    if verbose:
        print('Num Carto Studies:', num_carto_studies)

    # Process each patient's .pkl file
    for p, pickle_file in enumerate(patient_folders):
        if verbose:
            print(f'- Opening file ({p + 1}, {num_carto_studies}): {pickle_file}')

        # Generate save paths
        aux_str = os.path.basename(pickle_file).split('.')[0]
        aux_save_file_name = f'{aux_str}_features.parquet'
        aux_save_file_path = os.path.join(saving_path, aux_save_file_name)
        aux_file_exists = os.path.isfile(aux_save_file_path)

        # Skip existing files unless overwrite is enabled
        if aux_file_exists and not overwrite:
            print(f'- {aux_save_file_path}. File already exists. SKIP!')

        else:
            # Initialize DataFrames to store output
            features_data = pd.DataFrame(columns=['patient_id', 'map_name', 'site_id', 'DomCycleL', 'NLE0',
                                                  'rest', 'discrete', 'fragmentation', 'activity_index'])
            activity_data = pd.DataFrame()

            # Load patient data
            aux_data = pd.read_pickle(pickle_file).reset_index()
            aux_num_cf = len(aux_data)

            # Process each mapping site within the patient file
            for cf, cf_data in aux_data.iterrows():
                if verbose:
                    print(f'  - Processing CartoFinder data {cf + 1}/{aux_num_cf}')

                bipolar_signals = cf_data.BipolarSignals
                unipolar_signals = cf_data.CartoFinderData.Data_EGM
                subsampling = cf_data.Subsample
                dom_cycle = cf_data.CartoFinderData.DominantCycleLength

                # Extract features using signal analysis function
                entanglement, binary_activity, extended_activity_index, \
                    nleo, nleo_lpf, nleo_binary_activity, nleo_activity, \
                    robust_bipolar_lats, robust_bipolar_intervals, dominant_cycle, \
                    discrete_fragmented_activity, discrete_fragmented_activity_percentage, activity_index, \
                    extended_activity_index, interval_init_end_activity, entanglement_hybrid_intervals, \
                    entanglement_discrete_intervals = detectEntanglementRAISV2(
                        unipolar_signals, bipolar_signals, fs=fs/sampling
                    )

                # Organize extracted matrices into labeled DataFrames
                df_binary_activity = pd.DataFrame(binary_activity)
                df_binary_activity['Type'] = 'Binary'

                df_discrete_fragmented_activity = pd.DataFrame(discrete_fragmented_activity)
                df_discrete_fragmented_activity['Type'] = 'Discrete'

                # Build LATs matrix with binary indices for activation
                lats = np.zeros(bipolar_signals.shape, dtype=int)
                for i, indices in enumerate(robust_bipolar_lats):
                    lats[i, indices] = 1
                df_lats = pd.DataFrame(lats)
                df_lats['Type'] = 'LATs'

                df_entanglement = pd.DataFrame(entanglement)
                df_entanglement['Type'] = 'Entanglement'

                # Merge all feature matrices into a long-format DataFrame
                activity = pd.concat([df_binary_activity, df_discrete_fragmented_activity, df_lats, df_entanglement],
                                     axis=0)
                activity['patient_id'] = cf_data.patient_id
                activity['map'] = cf_data['map']
                activity['site_id'] = cf_data.site_id
                activity_data = pd.concat([activity_data, activity], axis=0)

                # Flatten and organize per-window summary features
                new_rows = pd.DataFrame({
                    'patient_id': cf_data.patient_id,
                    'map_name': cf_data['map'],
                    'site_id': cf_data.site_id,
                    'subsampling': subsampling,
                    'DomCycleL': dom_cycle,
                    'NLEO': nleo_activity.flatten().tolist(),
                    'rest': discrete_fragmented_activity_percentage[:, 0].flatten().tolist(),
                    'discrete': discrete_fragmented_activity_percentage[:, 1].flatten().tolist(),
                    'fragmentation': discrete_fragmented_activity_percentage[:, 2].flatten().tolist(),
                    'activity_index': activity_index.flatten().tolist(),
                    'activity_index_ext': extended_activity_index.flatten().tolist()
                })

                features_data = pd.concat([features_data, new_rows], ignore_index=True)

            # Save results to compressed parquet files
            activity_data.to_parquet(os.path.join(saving_path, f'{aux_str}_activity.parquet'), compression="zstd")
            features_data.to_parquet(aux_save_file_path, compression="zstd")
