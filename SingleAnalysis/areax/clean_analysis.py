from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import tqdm.auto as tqdm
import pickle, yaml, importlib
import matplotlib.pyplot as plt, seaborn as sns, matplotlib
from helper import singleglob, fastsearch, subarray_positions, get_param
import logging, beautifullogger

logger=logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)

session = Path("../Data/AreaXB602022-04-20_15-16-37")
song_fs = 32000
t_start = get_param("t_start")
t_end = get_param("t_end")
if not t_end:
    t_end=np.inf
song_dataset = xr.Dataset()
song_dataset["song"] = xr.DataArray(np.load(singleglob(session, "**/song.npy")).reshape(-1), dims="t")
song_dataset["t"] = np.arange(song_dataset["song"].size)/song_fs
song_dataset = song_dataset.sel(t=slice(t_start, t_end))

syb_dataset = xr.Dataset.from_dataframe(pd.read_csv(singleglob(session, "**/uncorrected_labels.csv"), sep=",", header=None, names=["uncorrected_start_index", "uncorrected_end_index", "syb_name"]))
syb_dataset["uncorrected_start"] = syb_dataset["uncorrected_start_index"]/song_fs
syb_dataset["uncorrected_end"] = syb_dataset["uncorrected_end_index"]/song_fs
syb_dataset = syb_dataset.where((syb_dataset["uncorrected_start"] >= t_start) & (syb_dataset["uncorrected_end"]<=t_end), drop=True)
syb_dataset = syb_dataset[["uncorrected_start", "uncorrected_end", "syb_name"]]


bua_fs=1000
neuro_dataset = xr.Dataset()
neuro_dataset["bua"] = xr.DataArray(np.load(singleglob(session, "**/CSC17*.npy")).reshape(-1), dims="t")
neuro_dataset["t"] = np.arange(neuro_dataset["bua"].size)/bua_fs
neuro_dataset = neuro_dataset.sel(t=slice(t_start, t_end))


# ============================================================================ #
# Handling song
# ============================================================================ #

song_dataset["amp"] = np.abs(song_dataset["song"]).rolling(t=int(song_fs*get_param("amp_window_size"))).mean()

# ============================================================================ #
# Adjusting syllables
# ============================================================================ #

syb_dataset["amp_threshold"] = xr.concat([
    song_dataset["amp"].sel(t=syb_dataset["uncorrected_start"], method="nearest"),
    song_dataset["amp"].sel(t=syb_dataset["uncorrected_end"], method="nearest")
], dim="is_start").quantile(get_param("amp_threshold_quantile")).item()


prev_uncorrected_end = syb_dataset["uncorrected_end"].shift(index=1)
next_uncorrected_start = syb_dataset["uncorrected_start"].shift(index=-1)

for dir, step in dict(start=-1, end=1).items():
    sp = xr.DataArray(song_dataset["t"].searchsorted(syb_dataset[f"uncorrected_{dir}"]), dims="index", coords=dict(index=syb_dataset["index"]))
    search = xr.apply_ufunc(fastsearch,
        song_dataset["amp"] < syb_dataset["amp_threshold"], 
        sp, step, 
        int(get_param("max_syllable_adjust")*song_fs), -1, input_core_dims=[["t"]] + [[]]*4
    , vectorize=False)

    tmp = xr.where(search >=0,  song_dataset["t"].isel(t=search).drop_vars("t"), syb_dataset[f"uncorrected_{dir}"])
    if dir=="start":
        syb_dataset[f"{dir}"] = xr.where(tmp < prev_uncorrected_end,prev_uncorrected_end, tmp)
    if dir=="end":
        syb_dataset[f"{dir}"] = xr.where(tmp > next_uncorrected_start,next_uncorrected_start, tmp)


# ============================================================================ #
# Selecting subsyllables
# ============================================================================ #

valid_positions = np.sort(np.unique(np.concatenate(
    [i+np.arange(len(motif)) for motif in get_param("motifs") for i in subarray_positions(syb_dataset["syb_name"].to_numpy().astype("U1"), np.array(list(motif)).astype("U1")) 
    ]
)))

valid_syllables = syb_dataset.isel(index=valid_positions)
positions = xr.DataArray(get_param("subsyllable_positions"), dims="syb_position", coords=dict(syb_position=np.arange(len(get_param("subsyllable_positions")))))
subsyllables_t = (syb_dataset["start"] + positions)
valid_syllables["subsyllable_t"] = subsyllables_t.where(subsyllables_t <= syb_dataset["end"])
valid_syllables["subsyllable_name"] = valid_syllables["syb_name"].astype(object).astype(str).astype(object) + valid_syllables["syb_position"].astype(object).astype(str).astype(object)
subsyllables = valid_syllables.stack(subindex=["index", "syb_position"]).reset_index("subindex")
subsyllables = subsyllables.where(subsyllables["subsyllable_t"], drop=True)
subsyllables = subsyllables[["syb_name", "subsyllable_t", "subsyllable_name"]]
print(subsyllables)


# ============================================================================ #
# Computing accoustic Features
# ============================================================================ #

def compute_entropy(a): 
    from scipy.signal import welch
    f, p = welch(a, nfft=512)
    if np.sum(p) == 0:
        return np.nan
    p /= np.sum(p)
    power_per_band_mat = p[p > 0]
    spectral_mat = -np.sum(power_per_band_mat * np.log2(power_per_band_mat))
    return spectral_mat

def compute_pitch(a):
    from pitch import processPitch
    return processPitch(a, song_fs, 1500, 2500)

# features["entropy"] = 
# features["pitch"] = xr.apply_ufunc(lambda a: compute_pitch(a), song.isel(t_song=sybpos+time_around), input_core_dims=[["win_index"]], vectorize=True)

accoustic_features = xr.Dataset()
accoustic_window = xr.DataArray(np.arange(2*int(song_fs*get_param("accoustic_window")/2)+1) - int(song_fs*get_param("accoustic_window")/2)/song_fs, dims="win_index")
accoustics_window_data = song_dataset["song"].sel(t=accoustic_window + subsyllables["subsyllable_t"], method="nearest")
print(accoustics_window_data)
accoustic_features["entropy"] = xr.apply_ufunc(compute_entropy, accoustics_window_data, input_core_dims=[["win_index"]], vectorize=True)
accoustic_features["pitch"] = xr.apply_ufunc(compute_pitch, accoustics_window_data, input_core_dims=[["win_index"]], vectorize=True)

print(accoustic_features)
# print(song_dataset)
# print(neuro_dataset)
# print(syb_dataset)
# print(valid_syllables)