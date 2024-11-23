# Effective stream must be greater than 1Hz 
import os
import pyxdf

import mne
from mne.preprocessing import ICA
from mne.filter import filter_data
from mne.viz import plot_topomap

import numpy as np

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import joblib
from pathlib import Path
from types import SimpleNamespace

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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


def extract_eeg(folder_name, fif_name=None):

    # Check if the FIF file already exists
    if os.path.exists(fif_name):
        print(f"File {fif_name} already exists. Loading data.")
        epochs = mne.read_epochs(fif_name, preload=True)
        
        # Distinguish between Go and NoGo based on event IDs
        go_epochs = epochs["Go"]
        nogo_epochs = epochs["NoGo"]
        return go_epochs, nogo_epochs

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
        trig = np.array(streams[i_trig]["time_series"][:, 0]).astype(int)
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
            ch_types=n_chan * ['eeg'] + ['stim']
        )
        raw_data = mne.io.RawArray(data_with_trigger, info)

        # Common average reference
        raw_data, ref_data = mne.set_eeg_reference(raw_data)

        # Manually create events array from trigger channel
        trig_diff = np.diff(trigger_resampled)
        events_indices = np.where(trig_diff != 0)[0] + 1
        events = np.column_stack((events_indices, np.zeros(len(events_indices)), trigger_resampled[events_indices].astype(int)))
        events = events.astype(int)

        # Create epochs for event_id 14
        epochs = mne.Epochs(
            raw_data,
            events,
            event_id={"Stimulus 1": 14},
            tmin=-1.0,
            tmax=4.0,
            baseline=(None, 0),
            preload=True
        )

        # Identify trials where event 14 follows a 7 (Go) or 6 (NoGo)
        go_trials = [i for i in range(len(events) - 1) if events[i, 2] == 7 and events[i + 1, 2] == 14]
        nogo_trials = [i for i in range(len(events) - 1) if events[i, 2] == 6 and events[i + 1, 2] == 14]
        go_trials = [epochs.selection.tolist().index(i + 1) for i in go_trials if i + 1 in epochs.selection]
        nogo_trials = [epochs.selection.tolist().index(i + 1) for i in nogo_trials if i + 1 in epochs.selection]
        go_epochs = epochs[go_trials]  # (n_trials x n_channels x n_samples)
        nogo_epochs = epochs[nogo_trials]
        all_go_epochs.append(go_epochs)
        all_nogo_epochs.append(nogo_epochs)

    all_go_epochs = mne.concatenate_epochs(all_go_epochs)
    all_nogo_epochs = mne.concatenate_epochs(all_nogo_epochs)

    # Map uppercase channel names to lowercase
    rename_mapping = {
        'FZ': 'Fz',
        'FCZ': 'FCz',
        'CZ': 'Cz',
        'CPZ': 'CPz',
        'PZ': 'Pz',
        'POZ': 'POz'
    }

    # Apply the renaming
    all_go_epochs.rename_channels(rename_mapping)
    all_nogo_epochs.rename_channels(rename_mapping)

    montage = mne.channels.make_standard_montage("standard_1020")
    all_go_epochs.set_montage(montage)
    all_nogo_epochs.set_montage(montage)

    # Combine and add annotations for differentiation
    all_epochs = mne.concatenate_epochs([all_go_epochs, all_nogo_epochs])
    all_epochs.events[:, 2] = np.hstack([
        np.full(len(all_go_epochs), 1),  # Label 'Go' epochs as 1
        np.full(len(all_nogo_epochs), 2)  # Label 'NoGo' epochs as 2
    ])
    all_epochs.event_id = {"Go": 1, "NoGo": 2}

    # Save the combined epochs
    all_epochs.save(fif_name, overwrite=True)
    print(f"Data saved to {fif_name}")

    return all_go_epochs, all_nogo_epochs



def plot_cnv_averages(go_epochs, nogo_epochs):
    # Compute trial averages for Cz channel
    go_avg = go_epochs.average(picks=['Cz'])
    nogo_avg = nogo_epochs.average(picks=['Cz'])

    # Get time values (shared for all epochs)
    times = go_avg.times
    
    # Plot average CNV
    plt.figure(figsize=(12, 6))
    plt.plot(times, go_avg.data[0], label=f'Go', alpha=0.7)
    plt.plot(times, nogo_avg.data[0], label=f'No-Go', linestyle='dashed', alpha=0.7)
    plt.axvline(x=0, color='k', linestyle='--', label='Stimulus 1 Onset')
    plt.title("Average CNV Signal for Go and No-Go Trials")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude for Cz Channel (µV)")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()


def plot_topoplots(go_epochs, nogo_epochs): 
    # Compute averages
    go_avg = go_epochs.average()
    nogo_avg = nogo_epochs.average()

    # Define time points
    time_points = [0.5, 1.5, 2.5, 3.5]
    time_indices = [np.argmin(np.abs(go_avg.times - t)) for t in time_points]

    # create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Go and No-Go at Different Time Points", fontsize=16)
    for i, t_idx in enumerate(time_indices):
        im, _ = plot_topomap(
            go_avg.data[:, t_idx],  # Data at the specific time index
            go_avg.info,
            axes=axes[0, i],
            show=False
        )
        axes[0, i].set_title(f"{time_points[i]}s (Go)", fontsize=10)

        im, _ = plot_topomap(
            nogo_avg.data[:, t_idx],  # Data at the specific time index
            nogo_avg.info,
            axes=axes[1, i],
            show=False
        )
        axes[1, i].set_title(f"{time_points[i]}s (No-Go)", fontsize=10)

    plt.show()
    
def preprocess(all_go_epochs, all_nogo_epochs):
    
    ''' apply temporal and spatial filters (and ICA) '''

    '''
    # [Optional] Artifact removal 
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

    # plot the effect of temporal and spatial filters
    # vis_filter(all_go_epochs, all_nogo_epochs, subj="all")

    # Bandpass filter
    all_go_epochs.filter(l_freq=0.1, h_freq=1)
    all_nogo_epochs.filter(l_freq=0.1, h_freq=1)

    # Spatial filter
    all_go_epochs.set_eeg_reference('average', projection=True)  # 'average' means CAR
    all_nogo_epochs.set_eeg_reference('average', projection=True)

    all_go_epochs.apply_proj()
    all_nogo_epochs.apply_proj()

    return all_go_epochs, all_nogo_epochs


def downsample_data(X_train, X_test, factor=2, pca=False, n_components=None):
    '''
    Downsamples the EEG data and applies PCA for dimensionality reduction.
    
    Parameters:
    - X: ndarray of shape (n_samples, n_channels, n_times)
    - factor: int, downsampling factor for the time dimension
    - n_components: int or float, number of principal components to retain 
                    or the variance ratio to retain (if float)
    
    Returns:
    - X_transformed: ndarray of shape (n_samples, n_channels_reduced, n_times_downsampled)
    '''
    # Downsample the time dimension
    X_train_downsampled = X_train[:, :, ::factor]
    X_test_downsampled = X_test[:, :, ::factor]
    
    n_samples_train, n_channels, n_times = X_train_downsampled.shape
    n_samples_test = X_test_downsampled.shape[0]

    if not pca:
        return X_train_downsampled, X_test_downsampled

    X_train_transformed = np.zeros((n_samples_train, n_components, n_times))
    X_test_transformed = np.zeros((n_samples_test, n_components, n_times))
    
    # Apply PCA to each time slice across channels
    for t in range(n_times):
        # Extract the data for all samples at time t
        time_slice_train = X_train_downsampled[:, :, t]  # Shape: (n_samples_train, n_channels)
        time_slice_test = X_test_downsampled[:, :, t]    # Shape: (n_samples_test, n_channels)

        # Standardize the data across samples using training data parameters
        scaler = StandardScaler()
        time_slice_train_standardized = scaler.fit_transform(time_slice_train)
        time_slice_test_standardized = scaler.transform(time_slice_test)

        # Apply PCA trained on training data
        pca_model = PCA(n_components=n_components)
        reduced_train = pca_model.fit_transform(time_slice_train_standardized)  # Shape: (n_samples_train, n_components)
        reduced_test = pca_model.transform(time_slice_test_standardized)        # Shape: (n_samples_test, n_components)

        # Assign reduced data back
        X_train_transformed[:, :, t] = reduced_train
        X_test_transformed[:, :, t] = reduced_test

    return X_train_transformed, X_test_transformed


def prepare_data(all_go_epochs_train, all_nogo_epochs_train, all_go_epochs_test, all_nogo_epochs_test, time_ds_factor=2, use_pca=False, n_components=20):
    """
    Training (Online Session 1) and Testing set (Offline Session 1 & 2).

    Prepares the training and test datasets by preprocessing, converting to NumPy arrays, 
    and creating labels for Go and NoGo epochs.

    Parameters:
    - all_go_epochs_train: Go epochs for training
    - all_nogo_epochs_train: NoGo epochs for training
    - all_go_epochs_test: Go epochs for testing
    - all_nogo_epochs_test: NoGo epochs for testing

    Returns:
    - X_train: Combined Go and NoGo data for training
    - y_train: Labels for training data
    - X_test: Combined Go and NoGo data for testing
    - y_test: Labels for testing data
    """
    # Convert training data to Numpy arrays and prepare labels
    X_go_train = all_go_epochs_train.get_data()  # Shape: (n_go_train_epochs, n_channels, n_times)
    X_nogo_train = all_nogo_epochs_train.get_data()  # Shape: (n_nogo_train_epochs, n_channels, n_times)

    # Convert test data to Numpy arrays and prepare labels
    X_go_test = all_go_epochs_test.get_data()  # Shape: (n_go_test_epochs, n_channels, n_times)
    X_nogo_test = all_nogo_epochs_test.get_data()  # Shape: (n_nogo_test_epochs, n_channels, n_times)

    # Create labels
    y_go_train = np.ones(X_go_train.shape[0], dtype=int)    # Label '1' for Go train epochs
    y_nogo_train = np.zeros(X_nogo_train.shape[0], dtype=int)  # Label '0' for NoGo train epochs
    y_go_test = np.ones(X_go_test.shape[0], dtype=int)      # Label '1' for Go test epochs
    y_nogo_test = np.zeros(X_nogo_test.shape[0], dtype=int)  # Label '0' for NoGo test epochs

    # Combine data
    X_train = np.concatenate((X_go_train, X_nogo_train), axis=0)
    y_train = np.concatenate((y_go_train, y_nogo_train), axis=0)
    X_test = np.concatenate((X_go_test, X_nogo_test), axis=0)
    y_test = np.concatenate((y_go_test, y_nogo_test), axis=0)

    # Shuffle training data
    np.random.seed(42)
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Downsample
    X_train, X_test = downsample_data(X_train, X_test, factor=time_ds_factor, pca=use_pca, n_components=n_components)

    # Flatten along the time axis
    n_samples_train, n_channels, n_times_train = X_train.shape
    n_samples_test, n_channels, n_times_test = X_test.shape

    X_train_flat = X_train.reshape(n_samples_train, -1)  # Shape: (n_samples, n_channels * n_times)
    X_test_flat = X_test.reshape(n_samples_test, -1)    # Shape: (n_samples, n_channels * n_times)

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    # Reshape back to original dimensions
    X_train = X_train_flat.reshape(n_samples_train, n_channels, n_times_train)
    X_test = X_test_flat.reshape(n_samples_test, n_channels, n_times_test)

    # Temporary feature
    X_train = X_train.mean(axis=2) 
    X_test = X_test.mean(axis=2)

    print("Combined train data shape:", X_train.shape)
    print("Combined train labels shape:", y_train.shape)
    print("Combined test data shape:", X_test.shape)
    print("Combined test labels shape:", y_test.shape)

    return X_train, y_train, X_test, y_test


def run_model(X, y, mdl='svm', n_splits = 4):
    """
    Run a model using k-fold cross-validation on the combined training and test data.

    Parameters:
    - X: np.ndarray, feature data (shape: [n_samples, n_features])
    - y: np.ndarray, target labels (shape: [n_samples])
    - model: sklearn-like model object with `fit` and `predict` methods
    - n_splits: int, number of folds for cross-validation

    Returns:
    - avg_accuracy: float, average accuracy across folds
    """

    if mdl == "svm":
        model = SVC(kernel='rbf', random_state=42)
    elif mdl == "rf":
        model = RandomForestClassifier(
            n_estimators=500,  # Number of trees in the forest
            max_depth=None,    # Maximum depth of each tree (None means nodes are expanded until all leaves are pure or until min_samples_split is reached)
            min_samples_split=2,  # Minimum number of samples required to split an internal node
            min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
            random_state=42,      # Ensures reproducibility
            n_jobs=-1,            # Use all processors for parallel computation
            class_weight=None     # Specify weights for classes if needed (e.g., for imbalanced datasets)
        )
    elif mdl == "lda":
        model = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, n_components=None)
    elif mdl == "xgb":
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # For binary classification
            n_estimators=500,            # Number of trees (iterations)
            max_depth=6,                 # Maximum depth of each tree
            learning_rate=0.3,           # Learning rate
            random_state=42
        )

    # Initialize k-fold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    confusion_matrices = []
    classification_reports = []
    best_model = None

    # Perform k-fold cross-validation
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        # Split data into training and validation sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model on the training fold
        model.fit(X_train, y_train)

        if best_model is None:
            best_model = model

        # Predict on the validation fold
        y_pred = model.predict(X_test)

        # Compute accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        if accuracy > accuracies[-1]:
            best_model = model

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

        # Compute classification report
        cr = classification_report(y_test, y_pred, output_dict=False)
        classification_reports.append(cr)

        # Display fold results
        print(f"\nFold {fold_idx + 1}")
        print(f"Fold Accuracy: {accuracy:.4f}")

    # Compute the average accuracy across all folds
    avg_accuracy = np.mean(accuracies)

    model_full_train = model.fit(X, y)

    # print(f"\nAverage accuracy across {n_splits} folds: {avg_accuracy:.2f}")
    return model_full_train, avg_accuracy, classification_reports, confusion_matrices


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy, "conf_matrix": conf_matrix, 'class_report' : class_report}


def feature_extraction(X):
    ''' extract significant features to improve model performance '''
    pass

def compute_empirical_chance(X, y, clf, n_iterations=100):
    from sklearn.model_selection import cross_val_score

    ''' compute chance level calculation '''
    print("running chance level ...")    
    chance_scores = []
    for i in range(n_iterations):

        print(f"iteration {i}")

        # Randomly permute labels
        y_permuted = np.random.permutation(y)

        scores = cross_val_score(clf, X, y_permuted, cv=5)

        chance_scores.append(np.mean(scores))
    
    empirical_chance_mean, empirical_chance_std = np.mean(chance_scores), np.std(chance_scores)
    # print(f"Empirical Chance Level: {empirical_chance_mean:.3f} ± {empirical_chance_std:.3f}")
    
    return empirical_chance_mean, empirical_chance_std

def vis_filter(all_go_epochs, all_nogo_epochs, subj):
    # Average the concatenated epochs
    go_evoked_no_filter = all_go_epochs.average()
    nogo_evoked_no_filter = all_nogo_epochs.average()

    # Apply bandpass filter
    go_evoked_bandpass = go_evoked_no_filter.copy().filter(l_freq=1, h_freq=40)
    nogo_evoked_bandpass = nogo_evoked_no_filter.copy().filter(l_freq=1, h_freq=40)

    # Apply bandpass + spatial filter (e.g., Laplacian or custom)
    go_evoked_bandpass_spatial = go_evoked_bandpass.copy().set_eeg_reference(ref_channels="average")
    nogo_evoked_bandpass_spatial = nogo_evoked_bandpass.copy().set_eeg_reference(ref_channels="average")

    filter_titles = ["No Filter", "Bandpass", "Bandpass + Spatial"]

    # Prepare subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

    # Plotting Go ERPs
    print("\nPlot Go ERP")
    axs[0, 0].plot(go_evoked_no_filter.times, go_evoked_no_filter.data[0], color="black")
    axs[0, 0].set_title(f"Go ({filter_titles[0]})")

    axs[0, 1].plot(go_evoked_bandpass.times, go_evoked_bandpass.data[0], color="black")
    axs[0, 1].set_title(f"Go ({filter_titles[1]})")

    axs[0, 2].plot(go_evoked_bandpass_spatial.times, go_evoked_bandpass_spatial.data[0], color="black")
    axs[0, 2].set_title(f"Go ({filter_titles[2]})")

    # Plotting NoGo ERPs
    print("\nPlot NoGo ERP")
    axs[1, 0].plot(nogo_evoked_no_filter.times, nogo_evoked_no_filter.data[0], color="black")
    axs[1, 0].set_title(f"NoGo ({filter_titles[0]})")

    axs[1, 1].plot(nogo_evoked_bandpass.times, nogo_evoked_bandpass.data[0], color="black")
    axs[1, 1].set_title(f"NoGo ({filter_titles[1]})")

    axs[1, 2].plot(nogo_evoked_bandpass_spatial.times, nogo_evoked_bandpass_spatial.data[0], color="black")
    axs[1, 2].set_title(f"NoGo ({filter_titles[2]})")

    # Add labels and formatting
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (μV)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # Add a baseline
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)  # Mark event onset

    plt.tight_layout()
    save_dir = "figures"
    plt.savefig(f"{save_dir}/{subj}_erp_comparison.png", dpi=300)  # Save the figure as PNG
    plt.show()
