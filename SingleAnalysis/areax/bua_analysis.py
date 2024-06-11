from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import tqdm.auto as tqdm
import pickle, yaml, importlib
import matplotlib.pyplot as plt, seaborn as sns, matplotlib
from helper import search, singleglob, fastsearch, subarray_positions
import logging, beautifullogger

logger=logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)

session = Path("../Data/AreaXB602022-04-20_15-16-37")
song_fs = 32000
slice_start=0
slice_end=None
# slice_start=3400*song_fs
slice_end=4000*song_fs
song = xr.DataArray(np.load(singleglob(session, "**/song.npy")).reshape(-1), dims="t_song")[slice_start:slice_end]
labels = pd.read_csv(singleglob(session, "**/uncorrected_labels.csv"), sep=",", header=None, names=["uncorrected_start_index", "uncorrected_end_index", "syb_name"])
labels["uncorrected_start_index"] = labels["uncorrected_start_index"] - slice_start
labels["uncorrected_end_index"] = labels["uncorrected_end_index"] - slice_start
labels = labels.loc[(labels["uncorrected_start_index"] >= 0) & (labels["uncorrected_end_index"] < song.size)]
label_threshold = 2*10**(-4)
song["t_song"] = np.arange(song.size)/song_fs
draw = True

# print(labels)
labels["prev_uncorrected_end"] = labels["uncorrected_end_index"].shift(1)
labels["next_uncorrected_start"] = labels["uncorrected_start_index"].shift(-1)
if (labels["uncorrected_start_index"] < labels["prev_uncorrected_end"]).any():
    logger.warning(f'Problem with initial labels at indices\n{labels.loc[labels["uncorrected_start_index"] < labels["uncorrected_end_index"].shift(1)]}\nAdjusting starts...')
    labels["uncorrected_start_index"] = np.where(labels["prev_uncorrected_end"] > labels["uncorrected_start_index"], labels["prev_uncorrected_end"], labels["uncorrected_start_index"]).astype(int)

print("loaded")

# print(labels)
amp = np.abs(song).rolling(t_song=int(song_fs/100)).mean()
amp_at_label_start = amp.sel(t_song=labels["uncorrected_start_index"].to_numpy()/song_fs, method="nearest").quantile(0.2).item()
label_threshold = amp_at_label_start

tqdm.tqdm.pandas(desc="Computing real start/ind indices")
labels["start_index"] = fastsearch((amp < label_threshold).to_numpy().reshape(1, -1), labels["uncorrected_start_index"].to_numpy().copy(), -1, int(song_fs/10), -1)
labels["end_index"] = fastsearch((amp < label_threshold).to_numpy().reshape(1, -1), labels["uncorrected_end_index"].to_numpy().copy(), 1, int(song_fs/10), -1)

# labels=labels.loc[(labels["start_index"] >= 0) & (labels["end_index"] >= 0)]
labels["start_index"] = np.where(labels["start_index"]<0, labels["uncorrected_start_index"], labels["start_index"])
labels["end_index"] = np.where(labels["end_index"]<0, labels["uncorrected_end_index"], labels["end_index"])

labels["prev_end"] = labels["end_index"].shift(1)
labels["next_start"] = labels["start_index"].shift(-1)
labels["start_index"] = np.where(labels["prev_uncorrected_end"] > labels["start_index"], labels["prev_uncorrected_end"], labels["start_index"]).astype(int)
labels["end_index"] = np.where(labels["next_uncorrected_start"] < labels["end_index"], labels["next_uncorrected_start"], labels["end_index"]).astype(int)


valid_positions = np.sort(np.unique(np.concatenate(
    [i+np.arange(len(motif)) for motif in [
        "abcd", "azd", "acd", "ab"] for i in subarray_positions(labels["syb_name"].to_numpy().astype("U1"), np.array(list(motif)).astype("U1")) 
    ]
)))

labels2 = labels.iloc[valid_positions]
labels2=labels2.loc[labels2["syb_name"] == "a"]
labels2["subsybsongpos"] = np.round(labels2["start_index"]+song_fs/100).astype(int)
features = pd.DataFrame()
features["subsybsongpos"] = labels2["subsybsongpos"]
features["amp"] = amp.isel(t_song=labels2["subsybsongpos"].to_numpy().astype(int))
time_around = xr.DataArray(np.arange(2*int(song_fs/200)+1) - int(song_fs/200), dims="win_index")
sybpos = xr.DataArray(labels2["subsybsongpos"].to_numpy(), dims="syb")

def compute_entropy(a): 
    from scipy.signal import welch
    f, p = welch(a, nfft=512)
    p /= np.sum(p)
    power_per_band_mat = p[p > 0]
    spectral_mat = -np.sum(power_per_band_mat * np.log2(power_per_band_mat))
    return spectral_mat

def compute_pitch(a):
    from pitch import processPitch
    return processPitch(a, song_fs, 1500, 2500)

features["entropy"] = xr.apply_ufunc(lambda a: compute_entropy(a), song.isel(t_song=sybpos+time_around), input_core_dims=[["win_index"]], vectorize=True)
features["pitch"] = xr.apply_ufunc(lambda a: compute_pitch(a), song.isel(t_song=sybpos+time_around), input_core_dims=[["win_index"]], vectorize=True)
features=features.set_index("subsybsongpos")

import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model
scaler = sklearn.preprocessing.StandardScaler()
rescaled_features = pd.DataFrame()
rescaled_features[features.columns] = scaler.fit_transform(features)

pca = sklearn.decomposition.PCA()
pca_feats = pca.fit_transform(rescaled_features)

bua_fs=1000
bua = xr.DataArray(np.load(singleglob(session, "**/CSC17*.npy")).reshape(-1), dims="t_bua")
bua["t_bua"] = np.arange(bua.size)/bua_fs - slice_start/song_fs
bua = bua.sel(t_bua = slice(song["t_song"].min().item(), song["t_song"].max().item()))

lag=20
syb_bua = pd.DataFrame()
syb_bua["subsybsongpos"] = labels2["subsybsongpos"]
syb_bua[f"bua_lag{lag}"] = bua.sel(t_bua=labels2["subsybsongpos"].to_numpy()/song_fs - lag*10**(-3), method="nearest").to_numpy()
model= sklearn.linear_model.LinearRegression()
model.fit(pca_feats, syb_bua[f"bua_lag{lag}"])
score=model.score(pca_feats, syb_bua[f"bua_lag{lag}"])
transformed = model.predict(pca_feats)

n_bootstrap=5000
syb_bua_bootstrapped = pd.DataFrame()
bt_positions=np.random.randint(syb_bua["subsybsongpos"].min(), syb_bua["subsybsongpos"].max(), size=syb_bua["subsybsongpos"].size*n_bootstrap)
syb_bua_bootstrapped["subsybsongpos"] = bt_positions
syb_bua_bootstrapped["bt_index"] = (np.arange(n_bootstrap).reshape(-1, 1)* np.ones((1, syb_bua["subsybsongpos"].size))).reshape(-1)

syb_bua_bootstrapped["bua"] = bua.sel(t_bua=bt_positions/song_fs, method="nearest").to_numpy()
tqdm.tqdm.pandas(desc="bootstrap")
bt = syb_bua_bootstrapped.groupby("bt_index").progress_apply(lambda d: sklearn.linear_model.LinearRegression().fit(pca_feats, d["bua"]).score(pca_feats, d["bua"]))
bt = bt.to_frame(name="score")
p_value = (bt["score"] > score).sum()/bt["score"].size

feat_models = pd.DataFrame()
feat_models["feat"] = [f for f in features.columns if not "subsybsongpos" in f]
feat_models["model"] = feat_models.apply(lambda row: sklearn.linear_model.LinearRegression().fit(syb_bua[f"bua_lag{lag}"].to_numpy().reshape(-1, 1), features[row["feat"]]), axis=1)
feat_models["score"] = feat_models.apply(lambda row: row["model"].score(syb_bua[f"bua_lag{lag}"].to_numpy().reshape(-1, 1), features[row["feat"]]), axis=1)
feat_models["transformed"] = feat_models.apply(lambda row: row["model"].predict(syb_bua[f"bua_lag{lag}"].to_numpy().reshape(-1, 1)), axis=1)
feat_models["bt_locs"] = [ np.random.randint(0, len(features.index), size=(n_bootstrap, len(features.index))) for _ in range(len(feat_models.index))]
feat_models["bt_score_dist"] = feat_models.progress_apply(lambda row: 
    np.array([
        sklearn.linear_model.LinearRegression().fit(syb_bua[f"bua_lag{lag}"].to_numpy().reshape(-1, 1), features[row["feat"]].iloc[row["bt_locs"][i]]).score(
            syb_bua[f"bua_lag{lag}"].to_numpy().reshape(-1, 1), features[row["feat"]].iloc[row["bt_locs"][i]]
        )
        for i in range(n_bootstrap)])
, axis=1)
feat_models["p_value"] = feat_models.apply(lambda row: np.sum(row["bt_score_dist"] > row["score"])/row["bt_score_dist"].size, axis=1)
feat_models = feat_models.set_index("feat")
features=features.reset_index()

if draw:
    f =  plt.figure(layout="tight")
    grid = matplotlib.gridspec.GridSpec(7, 12, figure=f)

    song_ax: plt.Axes = f.add_subplot(grid[0, :])
    spectrogram_ax: plt.Axes = f.add_subplot(grid[1, :])
    feats_ax: plt.Axes = f.add_subplot(grid[2, :])
    bua_ax: plt.Axes = f.add_subplot(grid[3, :])
    amp_ax_dist = f.add_subplot(grid[4, slice(0, 3)])
    entropy_ax_dist = f.add_subplot(grid[4, slice(3, 6)])
    pitch_ax_dist = f.add_subplot(grid[4, slice(6, 9)])
    bua_ax_dist = f.add_subplot(grid[4, slice(9, 12)])

    m_amp_ax_trs = f.add_subplot(grid[5, slice(0, 2)])
    m_entropy_ax_trs = f.add_subplot(grid[5, slice(2, 4)])
    m_pitch_ax_trs = f.add_subplot(grid[5, slice(4, 6)])
    ms_amp_ax_trs = f.add_subplot(grid[5, slice(6, 8)])
    ms_entropy_ax_trs = f.add_subplot(grid[5, slice(8, 10)])
    ms_pitch_ax_trs = f.add_subplot(grid[5, slice(10, 12)])
    m_bt_ax = f.add_subplot(grid[6, slice(0, 6)])
    amp_bt_ax = f.add_subplot(grid[6, slice(6, 8)])
    entropy_bt_ax = f.add_subplot(grid[6, slice(8, 10)])
    pitch_bt_ax = f.add_subplot(grid[6, slice(10, 12)])

    song_ax.sharex(spectrogram_ax)
    spectrogram_ax.sharex(feats_ax)
    feats_ax.sharex(bua_ax)

    amp_ax_dist.sharex(m_amp_ax_trs)
    # m_amp_ax_trs.sharex(amp_bt_ax)
    # m_entropy_ax_trs.sharex(entropy_bt_ax)
    entropy_ax_dist.sharex(m_entropy_ax_trs)
    pitch_ax_dist.sharex(m_pitch_ax_trs)
    # m_pitch_ax_trs.sharex(pitch_bt_ax)

    bua_ax_dist.sharex(ms_amp_ax_trs)
    ms_amp_ax_trs.sharex(ms_entropy_ax_trs)
    ms_entropy_ax_trs.sharex(ms_pitch_ax_trs)

    song.plot.line(ax=song_ax, zorder=1)
    amp.plot.line(ax=song_ax)
    song_ax.axhline(label_threshold, color="black", linestyle=":")


    song_ax.vlines(labels["start_index"]/song_fs, song.min().item(), song.max(), color="green", alpha=0.5)
    song_ax.vlines(labels["end_index"]/song_fs, song.min().item(), song.max(), color="red", alpha=0.5)
    song_ax.vlines(labels["uncorrected_start_index"]/song_fs, song.min().item(), song.max(), color="lightgreen", linestyle=":")
    song_ax.vlines(labels["uncorrected_end_index"]/song_fs, song.min().item(), song.max(), color="pink", linestyle=":")
    song_ax.scatter(labels2["subsybsongpos"].to_numpy()/song_fs,np.ones(labels2["subsybsongpos"].shape) * (song.max().item())/2, color="black")
    for _, row in labels.iterrows():
        song_ax.text((row["uncorrected_start_index"]+row["uncorrected_end_index"])/(2*song_fs), song.max()*0.8, row["syb_name"])
    for _, row in labels2.iterrows():
        spectrogram_ax.axvline(row["start_index"]/song_fs, color="green")
        spectrogram_ax.axvline(row["end_index"]/song_fs, color="red")
        spectrogram_ax.axvline(row["uncorrected_start_index"]/song_fs, color="lightgreen", linestyle=":")
        spectrogram_ax.axvline(row["uncorrected_end_index"]/song_fs, color="pink", linestyle=":")
        spectrogram_ax.axvline((row["subsybsongpos"] - int(song_fs/200))/song_fs, color="black", linestyle=":")
        spectrogram_ax.axvline((row["subsybsongpos"] + int(song_fs/200))/song_fs, color="black", linestyle=":")

    spectrogram_ax.specgram(song, Fs=song_fs, zorder=1)

    spectrogram_ax.scatter(features["subsybsongpos"]/song_fs, features["pitch"], color="orange", label="pitch")
    feats_ax.scatter(features["subsybsongpos"]/song_fs, features["entropy"], color="orange", label="entropy")
    feats_ax.legend()
    song_ax.scatter(features["subsybsongpos"]/song_fs, features["amp"], color="orange", zorder=2, label="amplitude")
    sns.histplot(features["amp"], ax=amp_ax_dist, bins=20, kde=True)
    sns.histplot(features["entropy"], ax=entropy_ax_dist, bins=20, kde=True)
    sns.histplot(features["pitch"], ax=pitch_ax_dist, bins=20, kde=True)
    sns.histplot(syb_bua[f"bua_lag{lag}"], ax=bua_ax_dist, bins=20, kde=True)

    bua.plot.line(ax=bua_ax)
    bua_ax.scatter(syb_bua["subsybsongpos"]/song_fs, syb_bua[f"bua_lag{lag}"], color="red", label=f"lag={lag}ms")
    bua_ax.scatter(syb_bua["subsybsongpos"]/song_fs - lag*10**(-3), syb_bua[f"bua_lag{lag}"], color="red", marker="+")
    bua_ax.legend()


    m_amp_ax_trs.scatter(features["amp"], transformed, color="red", marker="+", label="transformed")
    m_amp_ax_trs.scatter(features["amp"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+", label="real")
    m_amp_ax_trs.legend()

    m_entropy_ax_trs.scatter(features["entropy"], transformed, color="red", marker="+")
    m_entropy_ax_trs.scatter(features["entropy"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+")

    m_pitch_ax_trs.scatter(features["pitch"], transformed, color="red", marker="+")
    m_pitch_ax_trs.scatter(features["pitch"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+")
    
    sns.histplot(bt, x="score", ax=m_bt_ax, bins=20)
    m_bt_ax.axvline(score, color="red")
    m_bt_ax.text(score*1.05, m_bt_ax.get_ylim()[1]/2, f"p-value={p_value}", color="red")

    ms_entropy_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["entropy", "transformed"], color="red", marker="+")
    ms_entropy_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["entropy"], color="orange", marker="+")

    ms_amp_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["amp", "transformed"], color="red", marker="+")
    ms_amp_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["amp"], color="orange", marker="+")

    ms_pitch_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["pitch", "transformed"], color="red", marker="+")
    ms_pitch_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["pitch"], color="orange", marker="+")

    for feat, ax in {"amp": amp_bt_ax, "entropy": entropy_bt_ax, "pitch": pitch_bt_ax}.items():
        sns.histplot(pd.DataFrame(feat_models.loc[feat, "bt_score_dist"].reshape(-1, 1), columns=["score"]), x="score", ax=ax, bins=20)
        ax.axvline(feat_models.loc[feat, "score"], color="red")
        ax.text(feat_models.loc[feat, "score"]*1.05, ax.get_ylim()[1]/2, f'p-value={feat_models.loc[feat, "p_value"]}', color="red")

    plt.show()
