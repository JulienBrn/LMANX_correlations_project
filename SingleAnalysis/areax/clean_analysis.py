from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model
import tqdm.auto as tqdm
import pickle, yaml, importlib
import matplotlib.pyplot as plt, seaborn as sns, matplotlib
from helper import singleglob, fastsearch, subarray_positions, get_param
import logging, beautifullogger

logger=logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.WARNING)
rd_generator = np.random.default_rng()

session = Path("../Data/AreaXB602022-04-20_15-16-37")
song_fs = 32000
t_start = get_param("t_start")
t_end = get_param("t_end")
if not t_end:
    t_end=np.inf
song_dataset = xr.Dataset()
song_dataset = song_dataset.assign_coords(song_fs = song_fs)
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
neuro_dataset = neuro_dataset.assign_coords(neuro_fs=bua_fs)
neuro_dataset["bua"] = xr.DataArray(np.load(singleglob(session, "**/CSC17*.npy")).reshape(-1), dims="t").expand_dims(sensor=["CSC17"])

neurons=["ch11#1"]
spikes=[]
for n in neurons:
    ar = xr.DataArray(np.loadtxt(singleglob(session, f"**/{n}.txt")).reshape(-1), dims="spike")
    ar = ar.assign_coords(spike_num=("spike", np.arange(ar.size)), neuron=n)
    spikes.append(ar)

tmp_spikes = xr.concat(spikes, "spike")
neuro_dataset["t"] = np.arange(neuro_dataset["t"].size)/bua_fs
neuro_dataset = neuro_dataset.sel(t=slice(t_start, t_end))
neuro_dataset["spikes"] = tmp_spikes.where((tmp_spikes >=t_start) & (tmp_spikes <=t_end), drop=True)



# ============================================================================ #
# Handling song
# ============================================================================ #

def filter_song(sig, fs):
    import scipy.signal
    if fs > 2*get_param("max_song_f") :
        filter = scipy.signal.butter(4, [get_param("min_song_f"), get_param("max_song_f")], btype="bandpass", fs=fs, output="sos")
    else:
        logger.warning(f"Impossible to bandfilter song because song_fs = {fs}, falling back to highpassing")
        filter = scipy.signal.butter(4, get_param("min_song_f"), btype="highpass", fs=fs, output="sos")
    res = scipy.signal.sosfiltfilt(filter, sig)
    return res

song_dataset["filtered_song"] = xr.apply_ufunc(filter_song, song_dataset["song"], song_fs, input_core_dims=[["t"]]+ [[]], output_core_dims=[["t"]])
song_dataset["amp"] = np.abs(song_dataset["filtered_song"]).rolling(t=int(song_fs*get_param("amp_window_size"))).mean()

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
subsyllables = subsyllables.set_coords("subsyllable_name")
# print(subsyllables)


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


accoustic_features = xr.Dataset()
accoustic_window = xr.DataArray(np.arange(2*int(song_fs*get_param("accoustic_window")/2)+1) - int(song_fs*get_param("accoustic_window")/2)/song_fs, dims="win_index")
accoustics_window_data = song_dataset["filtered_song"].sel(t=accoustic_window + subsyllables["subsyllable_t"], method="nearest")
# print(accoustics_window_data)
accoustic_features["entropy"] = xr.apply_ufunc(compute_entropy, accoustics_window_data, input_core_dims=[["win_index"]], vectorize=True)
accoustic_features["pitch"] = xr.apply_ufunc(compute_pitch, accoustics_window_data, input_core_dims=[["win_index"]], vectorize=True)
accoustic_features["amp"] = song_dataset["amp"].sel(t=subsyllables["subsyllable_t"], method="nearest")

# print(accoustic_features)

subsyllables["accoustic_features"] = accoustic_features.to_array(dim="accoustic")

# print(subsyllables)

# ============================================================================ #
# Computing neuronal Features
# ============================================================================ #

def compute_ifr(spike_times):
    from elephant.statistics import instantaneous_rate
    from elephant import kernels
    from neo import SpikeTrain
    import quantities as pq
    kernel_sigma = get_param("spike_kernel_duration")*pq.s # decided by Arthur
    sampling_period_ifr = 1*pq.ms

    kernel = kernels.GaussianKernel(sigma=kernel_sigma)

    st = SpikeTrain(
        spike_times.to_numpy(),
        t_stop=neuro_dataset["t"].max().item()+1/bua_fs,
        t_start=neuro_dataset["t"].min().item(),
        units="s",
        sampling_rate=bua_fs * pq.Hz,
    )  # necessary to convert to use elephant

    ifr = instantaneous_rate(st, sampling_period=sampling_period_ifr, kernel=kernel).reshape(-1) # downsampled to 1000Hz
    # print(neuro_dataset["t"].size)
    # print(ifr.shape)
    if np.isnan(ifr).any():
        raise Exception("NA problems")
    return xr.DataArray(ifr, dims="t")

if len(neurons) >=2:
    neuron_ifrs = neuro_dataset["spikes"].groupby("neuron").map(compute_ifr)
    neuro_dataset["ifr"] = neuron_ifrs
elif len(neurons) == 1:
    neuro_dataset[f"ifr"] = compute_ifr(neuro_dataset["spikes"]).expand_dims(neuron=[neurons[0]])


lags = xr.DataArray(get_param("lags"), dims="lag", coords=dict(lag=get_param("lags")))
prepared_bua = neuro_dataset["bua"].assign_coords(sensor=neuro_dataset["sensor"] + "_bua").rename(sensor="neuro")
prepared_ifr = neuro_dataset["ifr"].assign_coords(neuron=neuro_dataset["neuron"]+"_ifr").rename(neuron="neuro")
neuro_features = xr.concat([prepared_bua, prepared_ifr], dim="neuro").sel(t=subsyllables["subsyllable_t"] -lags , method="nearest")
subsyllables["neuro_features"] = neuro_features
if subsyllables["neuro_features"].isnull().any():
    raise Exception("neuro feat extract na pb")
# print(neuro_features)
# print(subsyllables)

# ============================================================================ #
# Computing Models
# ============================================================================ #

models_dataset = xr.Dataset()

subsyllables["bootstrap_indices"] = subsyllables.groupby("subsyllable_name").map(
    lambda d: xr.DataArray(rd_generator.integers(0, d.sizes["subindex"], size=(d.sizes["subindex"], get_param("n_bootstrap"))), dims=["subindex", "bootstrap"]))

subsyllables["bootstrap_accoustic_features"] = subsyllables.groupby("subsyllable_name").map(lambda d: d["accoustic_features"].isel(subindex = d["bootstrap_indices"]))
subsyllables["bootstrap_neuro_features"] = subsyllables.groupby("subsyllable_name").map(lambda d: d["neuro_features"].isel(subindex = d["bootstrap_indices"]))


def lin_fitmodel(f, t):
    return sklearn.linear_model.LinearRegression().fit(f, t)

def predict(m, f):
    return m.predict(f)

for feat, target in [("accoustic", "neuro"), ("neuro", "accoustic")]:
    for prefix in ("", "bootstrap_"):
        models_dataset[f"{prefix}{feat}2{target}_model"] = xr.apply_ufunc(
            lin_fitmodel, subsyllables[f"{feat}_features"].groupby("subsyllable_name"), subsyllables[f"{prefix}{target}_features"].groupby("subsyllable_name"),
            input_core_dims=[["subindex", feat]] + [["subindex", target]],
            output_core_dims=[[]],
            vectorize=True
        )

        subsyllables[f"{prefix}predicted_{target}_features"] = xr.apply_ufunc(
            predict, models_dataset[f"{prefix}{feat}2{target}_model"], subsyllables[f"{feat}_features"].groupby("subsyllable_name"), 
            input_core_dims=  [[]] + [["subindex", feat]],
            output_core_dims=[["subindex", target]],
            vectorize=True
        )
        var_res= ((subsyllables[f"{prefix}predicted_{target}_features"] - subsyllables[f"{prefix}{target}_features"])**2).groupby("subsyllable_name").sum()
        var_to_mean = ((subsyllables[f"{prefix}{target}_features"].groupby("subsyllable_name").mean() - subsyllables[f"{prefix}{target}_features"])**2).groupby("subsyllable_name").sum()
        models_dataset[f"{prefix}{feat}2{target}_score"] = 1 - var_res / var_to_mean


# ============================================================================ #
# Finalizing
# ============================================================================ #


print(models_dataset)

with (session / "analysis_data.pkl").open("wb") as f:
    pickle.dump([song_dataset, syb_dataset, subsyllables, models_dataset, neuro_dataset], f)