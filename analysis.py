# Effective stream must be greater than 1Hz 
import mne
from mne.preprocessing import ICA

import numpy as np
import pyxdf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# Function to get stream indices based on type
def get_stream_indices(streams, type_data, type_trig):
    i_data, i_trig = -1, -1
    for i, stream in enumerate(streams):
        stream_type = stream['info']['type'][0].lower()
        effective_srate = float(stream['info']['effective_srate']) # effective stream rate
        print(effective_srate)
        if type_data in stream_type:
            i_data = i
        # if type_trig in stream_type and effective_srate > 0: ## use for sub-yy1
        if type_trig in stream_type: # use for all other subjects
            i_trig = i
    return i_data, i_trig

# Load XDF files iteratively from the folder py
folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-jh1/ses-S001/eeg"
# folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-tr1/ses-S001/eeg"
# folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-rn1/ses-S001/eeg"
# folder_name = "/home/minsu/Dropbox/Projects/VRCogTraining/Data/sub-yy1/ses-S001/eeg"

files = sorted([f for f in os.listdir(folder_name) if f.endswith(".xdf")])

all_go_epochs = []
all_nogo_epochs = []

for file in files:
    file_name = os.path.join(folder_name, file)
    print("\n----- ----- ----- ----- -----")
    print(f"File: {file_name}")
    streams, header = pyxdf.load_xdf(file_name)

    # Automatically determine the indices for each file
    i_data, i_trig = get_stream_indices(streams, 'eeg', 'stim')

    # Extract EEG data
    data = np.array(streams[i_data]["time_series"]).T # (39, N)
    sfreq = float(streams[i_data]["info"]["nominal_srate"][0])
    n_chan = 32
    data_times = np.array(streams[i_data]["time_stamps"]) # 5966.111482894273 ~ 6158.6251643241885

    # Convert to volts (if data is in microvolts)
    data /= 1e6

    # Extract trigger data
    ''' 
    1. 62 -> to start the game
    2. Go (7, 14, 30, 62, 158) or No Go (6, 14, 30, 158), each x 6
    Total of 55 (1 + 4 x 6 + 5 x 6)
    '''
    trig = np.array(streams[i_trig]["time_series"]).astype(int) 
    trig_times = np.array(streams[i_trig]["time_stamps"]) # 5974.31310947 ~ 6149.17227124

    """ Key Processing Step 
        - giving each data point a label by interpolating trigger data
    """

    # Interpolate trigger data to match the EEG data sampling rate
    interp_func = interp1d(trig_times, trig.flatten(), kind='nearest', fill_value="extrapolate")
    trigger_resampled = interp_func(np.arange(len(data[0])) / sfreq + data_times[0])

    # Append the trigger channel to the EEG data
    data_with_trigger = np.vstack((data[0:n_chan, :], trigger_resampled))

    # Create channel names variable
    channel_names = [ch['label'][0] for ch in streams[i_data]['info']['desc'][0]['channels'][0]['channel'] if ch['type'][0] == 'EEG'] + ['TRIGGER']

    # Create MNE info and RawArray for EEG data with trigger channel
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=['eeg'] * n_chan + ['stim']
    )
    raw_data = mne.io.RawArray(data_with_trigger, info)

    # Manually create events array from trigger channel
    trig_diff = np.diff(trigger_resampled) # when trigger changes from one to another = even takes place
    events_indices = np.where(trig_diff != 0)[0] + 1
    events = np.column_stack((events_indices, np.zeros(len(events_indices)), trigger_resampled[events_indices].astype(int)))
    events = events.astype(int)

    # Print and plot events
    unique_events = np.unique(events[:, 2])
    event_colors = dict(zip(unique_events, plt.cm.viridis(np.linspace(0, 1, len(unique_events)))))
    event_dict = {
        "Trial start - NoGo": 6,
        "Trial start - Go": 7,
        "Stimulus 1": 14,
        "Stimulus 2": 30,
        "Subject response": 62,
        "Feedback": 158,
    }
    color_dict = {event: color for event, color in zip(unique_events, plt.cm.viridis(np.linspace(0, 1, len(unique_events))))}
    # mne.viz.plot_events(events, sfreq=sfreq, color=color_dict, event_id=event_dict)

    # Create epochs for event_id 14 - when signal first shows up
    event_id = {'14': 14}
    # epochs = mne.Epochs(raw_data, events, event_id=event_id, tmin=-1.0, tmax=4.0, baseline=(None, 0), detrend=1, picks=['CZ'], preload=True)
    channels_to_pick = [ch for ch in channel_names if ch != 'TRIGGER']
    epochs = mne.Epochs(raw_data, events, event_id=event_id, tmin=-1.0, tmax=4.0, baseline=(None, 0), detrend=1, picks=channels_to_pick, preload=True)

    # Identify trials where event 14 follows a 7 (Go) or 6 (NoGo)
    go_trials = [i for i in range(len(events) - 1) if events[i, 2] == 7 and events[i + 1, 2] == 14]
    nogo_trials = [i for i in range(len(events) - 1) if events[i, 2] == 6 and events[i + 1, 2] == 14]

    # Convert to indices relative to epochs and ensure 1-based indexing
    go_trials = [epochs.selection.tolist().index(i + 1) for i in go_trials if i + 1 in epochs.selection]
    nogo_trials = [epochs.selection.tolist().index(i + 1) for i in nogo_trials if i + 1 in epochs.selection]

    # Select epochs for Go and NoGo
    go_epochs = epochs[go_trials]
    nogo_epochs = epochs[nogo_trials]
    # Append to all epochs list
    all_go_epochs.append(go_epochs)
    all_nogo_epochs.append(nogo_epochs)

# Concatenate all epochs
all_go_epochs = mne.concatenate_epochs(all_go_epochs)
all_nogo_epochs = mne.concatenate_epochs(all_nogo_epochs)

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

# Bandpass filter
all_go_epochs.filter(l_freq=0.1, h_freq=1)
all_nogo_epochs.filter(l_freq=0.1, h_freq=1)

# Spatial filter
all_go_epochs.set_eeg_reference('average', projection=True)  # 'average' means CAR
all_nogo_epochs.set_eeg_reference('average', projection=True)

all_go_epochs.apply_proj()
all_nogo_epochs.apply_proj()

# Average the concatenated epochs
go_evoked = all_go_epochs.average()
nogo_evoked = all_nogo_epochs.average()

subj = "jh"
# str_png = "no_filter"
# str_png = "bandpass"
str_png = "bandpass_spatial"

# Plot Go ERP
print("\nPlot Go ERP")
fig_go = go_evoked.plot(picks=['CZ'])
fig_go.suptitle('ERP for Event ID 14 Followed by Go', fontsize=16)
fig_go.savefig(f'{subj}_go_{str_png}.png', dpi=300)  # Save as PNG with high resolution

# Plot NoGo ERP
print("\nPlot NoGo ERP")
fig_nogo = nogo_evoked.plot(picks=['CZ'])
fig_nogo.suptitle('ERP for Event ID 14 Followed by NoGo', fontsize=16)
fig_nogo.savefig(f'{subj}_nogo_{str_png}.png', dpi=300)  # Save as PNG with high resolution

# Show plots
plt.show()