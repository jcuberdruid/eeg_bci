import mne
import numpy as np

raw = mne.io.read_raw_eeglab('../data/raw/sub-01/ses-S2/eeg/oneBACK.set', preload=True)
events, event_id = mne.events_from_annotations(raw)
sampling_rate = 500  
durations = np.diff(events[:, 0], append=events[-1, 0]) / sampling_rate

code_dict = {
    '6111': ' Block Start',
    '6112': ' Block End',
    '6121': ' Normal Trial Onset',
    '6122': ' Hit Trial Onset',
    '6123': ' Conflict Trial Onset',
    '6131': ' Error Response',
    '6132': ' Correct Response',
    '6133': ' Conflict Error',
    '611': ' End'
}


for event, duration in zip(events, durations):
    description = [k for k, v in event_id.items() if v == event[2]][0]

    if description == 'boundary':
        printDesc = 'boundary'
    else:
        printDesc = code_dict[description]

    print(f"Event ({printDesc}): {duration:.3f} seconds")
