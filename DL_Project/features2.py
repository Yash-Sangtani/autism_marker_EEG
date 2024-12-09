import fix_pyrqa
import pandas as pd
import logging
import numpy as np
import pywt
import pyrqa
import nolds
import os
import mne
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from concurrent.futures import ProcessPoolExecutor
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RPComputation

# Setting up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_rqa_features(data, embedding_dimension=3, time_delay=1, radius=0.1, minimum_line_length=2):
    """
    Extract Recurrence Quantitative Analysis (RQA) features from EEG signal.
    
    Parameters:
    -----------
    data : ndarray
        EEG signal for a single channel (timepoints).
    embedding_dimension : int
        Embedding dimension for the phase space.
    time_delay : int
        Time delay for embedding.
    radius : float
        Radius for defining recurrences.
    minimum_line_length : int
        Minimum line length for diagonal line analysis.
    
    Returns:
    --------
    dict
        RQA features including RR, DET, LAM, L_max, L_entr, L_mean, and TT.
    """


def calculate_rqa_features(data, embedding_dimension, time_delay, radius, minimum_line_length):
    # Create TimeSeries object
    time_series = TimeSeries(data, 
                             embedding_dimension=embedding_dimension, 
                             time_delay=time_delay)

    # Configure settings
    settings = Settings(time_series,
                        neighbourhood=FixedRadius(radius),
                        metric=EuclideanMetric)

    # Compute recurrence plot
    computation = RPComputation.create(settings)
    recurrence_matrix = computation.run()

    # Calculate RQA features manually from the recurrence matrix
    total_points = recurrence_matrix.shape[0] * recurrence_matrix.shape[1]
    
    # Recurrence Rate (RR)
    rr = np.sum(recurrence_matrix) / total_points

    # Determinism (DET)
    diagonal_lines = np.sum(np.sum(recurrence_matrix, axis=1) >= minimum_line_length)
    det = diagonal_lines / np.sum(recurrence_matrix)

    # Laminarity (LAM)
    vertical_lines = np.sum(np.sum(recurrence_matrix, axis=0) >= minimum_line_length)
    lam = vertical_lines / np.sum(recurrence_matrix)

    # Maximum Diagonal Line Length
    l_max = np.max(np.sum(np.triu(recurrence_matrix), axis=1))

    # Entropy of Diagonal Lines
    diagonal_lengths = np.sum(np.triu(recurrence_matrix), axis=1)
    diagonal_lengths = diagonal_lengths[diagonal_lengths > 0]
    diagonal_prob = diagonal_lengths / np.sum(diagonal_lengths)
    l_entr = -np.sum(diagonal_prob * np.log(diagonal_prob))

    # Mean Diagonal Line Length
    l_mean = np.mean(diagonal_lengths)

    # Trapping Time (approximation)
    tt = np.max(np.sum(recurrence_matrix, axis=0))

    # Return features
    rqa_features = {
        'RR': rr,
        'DET': det,
        'LAM': lam,
        'L_max': l_max,
        'L_entr': l_entr,
        'L_mean': l_mean,
        'TT': tt
    }

    return rqa_features

def extract_complexity_features(data):
    """
    Extract Sample Entropy and Detrended Fluctuation Analysis (DFA) features.

    Parameters:
    -----------
    data : ndarray
        EEG signal for a single channel (timepoints).

    Returns:
    --------
    dict
        Complexity features including Sample Entropy and DFA.
    """
    try:
        sample_entropy = nolds.sampen(data)
        dfa = nolds.dfa(data)

        return {
            'sample_entropy': sample_entropy,
            'dfa': dfa
        }
    except Exception as e:
        logging.error(f"Error calculating complexity features: {e}")
        return {
            'sample_entropy': np.nan,
            'dfa': np.nan
        }

def apply_wavelet_decomposition(data, wavelet='db4'):
    """
    Decompose EEG signal using Daubechies (db4) wavelet and return power of six frequency bands.

    Parameters:
    -----------
    data : ndarray
        EEG signal for a single channel (timepoints).
    wavelet : str
        The wavelet to be used (default is 'db4').

    Returns:
    --------
    dict
        Power of six frequency bands after wavelet decomposition.
    """
    # Perform wavelet decomposition (db4) on the signal
    coeffs = pywt.wavedec(data, wavelet)
    
    # Define six frequency bands for power estimation
    band_powers = {}
    band_powers['band_1'] = np.sum(np.abs(coeffs[0]))  # Approximation coefficients (low freq)
    band_powers['band_2'] = np.sum(np.abs(coeffs[1]))  # Detail coefficients (higher freq)
    band_powers['band_3'] = np.sum(np.abs(coeffs[2]))  # Detail coefficients
    band_powers['band_4'] = np.sum(np.abs(coeffs[3]))  # Detail coefficients
    band_powers['band_5'] = np.sum(np.abs(coeffs[4]))  # Detail coefficients
    band_powers['band_6'] = np.sum(np.abs(coeffs[5]))  # Detail coefficients (highest freq)

    return band_powers

def extract_features_from_eeg(data, channel_name, embedding_dimension=3, time_delay=1, radius=0.1, minimum_line_length=2):
    """
    Extract features from EEG data for a single channel.

    Parameters:
    -----------
    data : ndarray
        EEG signal for a single channel (timepoints).
    channel_name : str
        The name of the channel.
    embedding_dimension : int
        Embedding dimension for phase space.
    time_delay : int
        Time delay for embedding.
    radius : float
        Radius for defining recurrences.
    minimum_line_length : int
        Minimum line length for diagonal line analysis.

    Returns:
    --------
    dict
        Features for the given channel including wavelet decomposition, RQA, SampEn, and DFA.
    """
    features = {}
    features.update(apply_wavelet_decomposition(data))  # Power of frequency bands
    rqa_features = extract_rqa_features(data, embedding_dimension, time_delay, radius, minimum_line_length)
    features.update(rqa_features)
    complexity_features = extract_complexity_features(data)
    features.update(complexity_features)
    features['channel'] = channel_name
    return features

def load_and_preprocess_eeg(participant_id, path):
    """
    Load and preprocess EEG data for a participant.
    
    Parameters:
    -----------
    participant_id : str
        Identifier for the participant.
    path : str
        Path to the EEG dataset.
    
    Returns:
    --------
    dict
        Extracted features for all channels.
    """
    try:
        logging.info(f"Loading EEG data for participant: {participant_id}")
        file_path = os.path.join(path, f'{participant_id}_Resting.set')
        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage)

        # Only retain the channels available in the data
        common_channels = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
        available_channels = [ch for ch in common_channels if ch in raw_data.info['ch_names']]
        
        raw_data.pick_channels(available_channels)
        
        # Resample data to 512 Hz if necessary
        target_sfreq = 512
        if raw_data.info['sfreq'] != target_sfreq:
            raw_data.resample(target_sfreq)
        
        raw_data.filter(1, 40, fir_design='firwin')

        # Extract features from all channels
        data = raw_data.get_data()
        all_features = {}
        for i, channel_name in enumerate(raw_data.ch_names):
            all_features[channel_name] = extract_features_from_eeg(data[i], channel_name)

        logging.info(f"Preprocessing completed for participant: {participant_id}")
        return all_features

    except Exception as e:
        logging.error(f"Error processing data for participant {participant_id}: {e}")
        return None

def save_features_to_csv(participant_id, all_features, output_path):
    """
    Save extracted features to a CSV file for a participant.

    Parameters:
    -----------
    participant_id : str
        Identifier for the participant.
    all_features : dict
        Extracted features for all channels.
    output_path : str
        Directory path to save the CSV file.
    """
    logging.info(f"Saving features for participant: {participant_id}.")
    all_features_list = []
    
    for channel_name, features in all_features.items():
        row = {'participant_id': participant_id, 'channel': channel_name}
        row.update(features)
        all_features_list.append(row)
    
    df = pd.DataFrame(all_features_list)
    csv_path = os.path.join(output_path, f'{participant_id}_features.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Features saved to {csv_path}.")

def combine_all_features(output_path, combined_csv_path):
    """
    Combine features of all participants into a single CSV file.

    Parameters:
    -----------
    output_path : str
        Path where individual participant feature files are stored.
    combined_csv_path : str
        Path to save the combined CSV file.
    """
    logging.info("Combining all participant features into one CSV file.")
    feature_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('_features.csv')]
    combined_data = pd.concat([pd.read_csv(file) for file in feature_files], ignore_index=True)
    combined_data.to_csv(combined_csv_path, index=False)
    logging.info(f"Combined features saved to {combined_csv_path}.")

def process_participants(participant_ids, data_path, output_path):
    """
    Process all participants to extract features and save to CSV.

    Parameters:
    -----------
    participant_ids : list
        List of participant IDs to process.
    data_path : str
        Path to the EEG data files.
    output_path : str
        Path to save the output CSV files.
    """
    all_features = load_and_preprocess_eeg(participant_ids, data_path)

    # for participant_id in participant_ids:
    #     all_features = load_and_preprocess_eeg(participant_id, data_path)
    if all_features:
        save_features_to_csv(participant_ids, all_features, output_path)
    
    # Combine all individual participant features into one CSV
    combine_all_features(output_path, os.path.join(output_path, 'features2_combined.csv'))


def main():
    # Specify participant IDs, data path, and output path
    participant_ids = [
        'ASD1', 'ASD2', 'ASD3', 'ASD4', 'ASD5', 'ASD6', 'ASD7', 'ASD8', 'ASD9', 'ASD10', 'ASD11',
        'ASD12', 'ASD13', 'ASD14', 'ASD15', 'ASD16', 'ASD17', 'ASD18', 'ASD19', 'ASD20', 'ASD21',
        'ASD22', 'P51', 'ASD24', 'ASD25', 'ASD26', 'ASD27', 'ASD28', 'ASD29', 'P1', 'P5', 'P6',
        'P9', 'P10', 'P12', 'P16', 'P17', 'P18', 'P20', 'P24', 'P25', 'P26', 'P29', 'P31', 'P32',
        'P37', 'P38', 'P41', 'P42', 'P43', 'P44', 'P52', 'P53', 'P54', 'P56', 'P60'
    ]

    data_path = 'D:/SEMESTER5/DL/Project/autism_marker_EEG/Aging/'
    output_path = 'D:/SEMESTER5/DL/Project/autism_marker_EEG/Aging/features'
    combined_csv_path = 'D:/SEMESTER5/DL/Project/autism_marker_EEG/features2_combined.csv'
    
    os.makedirs(output_path, exist_ok=True)

        # Parallelize the processing of participants
    logging.info("Starting parallel processing of participants.")
    with ProcessPoolExecutor() as executor:
        executor.map(process_participants, participant_ids, [data_path] * len(participant_ids),
                     [output_path] * len(participant_ids))

    logging.info("All participants processed. Combining features.")
    # Combine all participant features into one CSV
    combine_all_features(output_path, combined_csv_path)

if __name__ == "__main__":
    main()
