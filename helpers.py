# Effective stream must be greater than 1Hz 
import os
import pyxdf

import mne
from mne.preprocessing import ICA
from mne.filter import filter_data

import numpy as np

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import joblib
from pathlib import Path
from types import SimpleNamespace

# set up parameters
eeg_stream_name = "eegoSports 000103"
n_chan = 32

# Function to get stream indices based on name
def get_stream_indices(streams, name_data, name_trig):
    i_data, i_trig = -1, -1
    for i, stream in enumerate(streams):
        stream_name = stream['info']['name'][0]
        if name_data in stream_name:
            i_data = i
        if name_trig in stream_name:
            i_trig = i
    return i_data, i_trig

def extract_eeg(folder_name):
    
    files = [f for f in os.listdir(folder_name) if f.endswith(".xdf")]

    all_go_epochs = []
    all_nogo_epochs = []

    for file in files:
        # Load XDF files iteratively from the folder
        file_name = os.path.join(folder_name, file)
        streams, header = pyxdf.load_xdf(file_name)
        i_data, i_trig = get_stream_indices(streams, eeg_stream_name, 'UnityTriggerStream')

        # Extract EEG data
        data = np.array(streams[i_data]["time_series"]).T
        sfreq = float(streams[i_data]["info"]["nominal_srate"][0])
        data_times = np.array(streams[i_data]["time_stamps"])

        # Extract event markers and interpolate
        trig = np.array(streams[i_trig]["time_series"][:,0]).astype(int)
        trig_times = np.array(streams[i_trig]["time_stamps"])
        interp_func = interp1d(trig_times, trig.flatten(), kind='nearest', fill_value="extrapolate")
        trigger_resampled = interp_func(np.arange(len(data[0])) / sfreq + data_times[0])

        # Append the trigger channel to the EEG data
        data_with_trigger = np.vstack((data[0:n_chan, :], trigger_resampled))
        channel_names = [ch['label'][0] for ch in streams[i_data]['info']['desc'][0]['channels'][0]['channel'] if ch['type'][0] == 'EEG'] + ['TRIGGER']

        # Create MNE info and RawArray for EEG data with trigger channel
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=sfreq,
            ch_types=n_chan*['eeg'] + ['stim']
        )
        raw_data = mne.io.RawArray(data_with_trigger, info)

        # Common average reference
        raw_data, ref_data = mne.set_eeg_reference(raw_data)
        
        # Manually create events array from trigger channel
        trig_diff = np.diff(trigger_resampled)
        events_indices = np.where(trig_diff != 0)[0] + 1
        events = np.column_stack((events_indices, np.zeros(len(events_indices)), trigger_resampled[events_indices].astype(int)))
        events = events.astype(int)
        
        # Plot events
        unique_events = np.unique(events[:, 2])
        event_colors = dict(zip(unique_events, plt.cm.viridis(np.linspace(0, 1, len(unique_events)))))
        event_dict = {
            "Trial start - NoGo": 6,
            "Trial start - Go": 7,
            "Stimulus 1": 14,
            "Stimulus 2": 30,
            "Subject response": 62,
            "Feedback": 158,
            "Classification - NoGo": 160,
            "Classification - Go": 170
        }
        color_dict = {event: color for event, color in zip(unique_events, plt.cm.viridis(np.linspace(0, 1, len(unique_events))))}
        # mne.viz.plot_events(events, sfreq=sfreq, color=color_dict, event_id=event_dict, on_missing='warn')
        
        # Create epochs for event_id 14
        epochs = mne.Epochs(
            raw_data,
            events,
            event_id={"Stimulus 1": 14},
            tmin=-1.0,
            tmax=4.0,
            baseline=(None, 0),
            preload=True)

        # Identify trials where event 14 follows a 7 (Go) or 6 (NoGo)
        go_trials = [i for i in range(len(events) - 1) if events[i, 2] == 7 and events[i + 1, 2] == 14]
        nogo_trials = [i for i in range(len(events) - 1) if events[i, 2] == 6 and events[i + 1, 2] == 14]
        go_trials = [epochs.selection.tolist().index(i + 1) for i in go_trials if i + 1 in epochs.selection]
        nogo_trials = [epochs.selection.tolist().index(i + 1) for i in nogo_trials if i + 1 in epochs.selection]
        go_epochs = epochs[go_trials] # (n_trials x n_channels x n_samples)
        nogo_epochs = epochs[nogo_trials]
        all_go_epochs.append(go_epochs)
        all_nogo_epochs.append(nogo_epochs)

    all_go_epochs = mne.concatenate_epochs(all_go_epochs)
    all_nogo_epochs = mne.concatenate_epochs(all_nogo_epochs)

    return all_go_epochs, all_nogo_epochs

def preprocess(all_go_epochs, all_nogo_epochs):
    '''
    # Artifact removal 
    # Fit ICA on the epochs (this will work best if you have a good amount of data)
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(all_go_epochs)  # Fit ICA on Go epochs
    ica.fit(all_nogo_epochs)  # Fit ICA on NoGo epochs

    # Step 2: Inspect ICA components and exclude artifact components (if needed)
    ica.plot_components()  # Visualize the components to identify artifacts (e.g., eye blinks)
    ica.exclude = [0, 3]  # Example: exclude components 0 and 3 (eye movement/artifacts)

    # Apply ICA to the data to remove the artifact components
    ica.apply(all_go_epochs)  # Apply ICA to remove artifacts from Go epochs
    ica.apply(all_nogo_epochs)  # Apply ICA to remove artifacts from NoGo epochs

    print("ICA applied.")
    '''

    ''' apply temporal and spatial filters '''
    
    # Bandpass filter
    all_go_epochs.filter(l_freq=0.1, h_freq=1)
    all_nogo_epochs.filter(l_freq=0.1, h_freq=1)

    # Spatial filter
    all_go_epochs.set_eeg_reference('average', projection=True)  # 'average' means CAR
    all_nogo_epochs.set_eeg_reference('average', projection=True)

    all_go_epochs.apply_proj()
    all_nogo_epochs.apply_proj()

    return all_go_epochs, all_nogo_epochs


def downsample_data(X, factor=2):
    return X[:, :, ::factor]

def prepare_data(all_go_epochs_train, all_nogo_epochs_train, all_go_epochs_test, all_nogo_epochs_test):
    
    """
    Training (Online Session 1) and Testing set (Offline Sessino 1 & 2).

    Prepares the training and test datasets by preprocessing, converting to NumPy arrays, 
    and creating labels for Go and NoGo epochs.

    Parameters:
    - all_go_epochs_train: Go epochs for training
    - all_nogo_epochs_train: NoGo epochs for training
    - all_go_epochs_test: Go epochs for testing
    - all_nogo_epochs_test: NoGo epochs for testing
    - preprocess: preprocessing function to be applied to each epoch set
    
    Returns:
    - X_train: Combined Go and NoGo data for training
    - y_train: Labels for training data
    - X_test: Combined Go and NoGo data for testing
    - y_test: Labels for testing data
    """

    # Convert training data to Numpy arrays and prepare labels
    X_go_train = all_go_epochs_train.get_data()      # Shape: (n_go_train_epochs, n_channels, n_times)
    X_nogo_train = all_nogo_epochs_train.get_data()  # Shape: (n_nogo_train_epochs, n_channels, n_times)

    # Convert test data to Numpy arrays and prepare labels
    X_go_test = all_go_epochs_test.get_data()        # Shape: (n_go_test_epochs, n_channels, n_times)
    X_nogo_test = all_nogo_epochs_test.get_data()    # Shape: (n_nogo_test_epochs, n_channels, n_times)

    # Create labels for train and test sets
    y_go_train = np.ones(X_go_train.shape[0], dtype=int)    # Label '1' for Go train epochs
    y_nogo_train = np.zeros(X_nogo_train.shape[0], dtype=int) # Label '0' for NoGo train epochs
    y_go_test = np.ones(X_go_test.shape[0], dtype=int)      # Label '1' for Go test epochs
    y_nogo_test = np.zeros(X_nogo_test.shape[0], dtype=int) # Label '0' for NoGo test epochs

    # Combine train data
    X_train = np.concatenate((X_go_train, X_nogo_train), axis=0)  # Shape: (n_total_train_epochs, n_channels, n_times)
    y_train = np.concatenate((y_go_train, y_nogo_train), axis=0)  # Shape: (n_total_train_epochs,)

    # Shuffle training data
    np.random.seed(42)
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]


    # Combine test data
    X_test = np.concatenate((X_go_test, X_nogo_test), axis=0)     # Shape: (n_total_test_epochs, n_channels, n_times)
    y_test = np.concatenate((y_go_test, y_nogo_test), axis=0)     # Shape: (n_total_test_epochs,)
    
    # Standardize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Downsample
    X_train = downsample_data(X_train, factor=4)
    X_test = downsample_data(X_test, factor=4)

    # temporary feature
    X_train = X_train.mean(axis=2) 
    X_test = X_test.mean(axis=2)

    print("Combined train data shape:", X_train.shape)
    print("Combined train labels shape:", y_train.shape)
    print("Combined test data shape:", X_test.shape)
    print("Combined test labels shape:", y_test.shape)
    
    return X_train, y_train, X_test, y_test

def feature_extraction(X):
    ''' extract significant features to improve model performance '''
    pass

def chance_level(X, y):
    ''' compute chance level calculation '''
    pass    