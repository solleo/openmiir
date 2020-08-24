__author__ = 'sgkim'

import sys, os
from mne.preprocessing import read_ica
from mne.io import read_raw_fif
from mne import pick_types

## Read system arguments:
if len(sys.argv) < 5:
    print("""
    USAGE:
    $ python applyica.py <SubjectID> <DirectoryRaw> <DirectoryICA> <DirectoryProc>
    
    EXAMPLE:
    $ python applyica.py P01 /media/sgk/Ext4_6TB/externaldata/openmiir/eeg/raw/ /media/sgk/Ext4_6TB/externaldata/openmiir/eeg/ica/  /media/sgk/Ext4_6TB/externaldata/openmiir/eeg/proc/
    """)
    sys.exit()

## CHECK FILES/PATHS:
sep = os.path.sep
sub = sys.argv[1]
fn_raw = f"{sys.argv[2]}{sep}{sub}-raw.fif"
fn_ica = f"{sys.argv[3]}{sep}{sub}-100p_64c-ica.fif"
fn_out = f"{sys.argv[4]}{sep}{sub}-bpf-ica-raw.fif"
print(f'INPUT RAW = {fn_raw}')
assert os.path.isfile(fn_raw), f"File [{fn_raw}] not found!"
print(f'INPUT ICA = {fn_ica}')
assert os.path.isfile(fn_ica), f"File [{fn_ica}] not found!"
print(f'OUTPUT PROC = {fn_out}')
assert os.path.isdir(sys.argv[4]), f"Directory [{sys.argv[4]}] not found!"

## CHECK DATA:
raw = read_raw_fif(fn_raw, preload=True)
#print(f"BAD CHANNELS = {raw.info['bads']}")
print(raw.info)

# P11+: additional 2 EOGs are incorrectly marked as EEG:
ch_kinds = [raw.info['chs'][i]['kind'] for i in range(len(raw.info['chs']))]
if ch_kinds.count(2) > 64:   # too many eeg(kind=2) channels!
    print(F"__WARNING__ Too many {ch_kinds.count(2)} 'eeg' channles are found! (should be 64). Manually correcting them.")
    raw.info['chs'][68]['kind'] = 202
    raw.info['chs'][69]['kind'] = 202

## Now, let's redo the pipeline:
# bad channel interpolation
raw.interpolate_bads(origin=[0, 0, 0])

# Bandpass filtering 0.5-30 Hz (FFT?)
eeg_pick = pick_types(raw.info, meg=False, eeg=True,
                          eog=False, stim=False, exclude=[])
raw.filter(0.5, 30, picks=eeg_pick, filter_length='40s',
                   l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft',
                   n_jobs=4, verbose=True)

## READ & APPLY ICA matrix
ica = read_ica(fn_ica)
print(f"BAD COMPONENTS = {ica.exclude}")
ica.apply(raw, exclude=ica.exclude)

raw.save(fname=fn_out)
