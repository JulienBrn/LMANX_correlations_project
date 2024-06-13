from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model
import tqdm.auto as tqdm
import pickle, yaml, importlib
import matplotlib.pyplot as plt, seaborn as sns, matplotlib
from helper import singleglob, fastsearch, subarray_positions, get_param
import logging, beautifullogger
from typing import List

with Path("./plot_params.yaml").open("r") as f:
        params = yaml.safe_load(f)

with Path(params["data_path"]).open("rb") as f:
    d: List[xr.Dataset] = pickle.load(f)
    [song_dataset, syb_dataset, subsyllables, models_dataset, neuro_dataset] = d

selected_syb = syb_dataset.where(syb_dataset["syb_name"] == params["subsyllable_name"][:1], drop=True)
selected_subsyb = subsyllables.where(subsyllables["subsyllable_name"]==params["subsyllable_name"], drop=True)
n_acoustics = selected_subsyb.sizes["accoustic"]
n_neuro = selected_subsyb.sizes["neuro"]
models_dataset = models_dataset.sel(model_name=params["model_name"], scoring_method=params["scoring_method"])

def draw_raw_song(ax: plt.Axes):
     song_dataset["song"].drop_vars("song_fs").plot.line(ax=ax, label="raw_song", color="lightblue")
     ax.vlines(syb_dataset["uncorrected_start"], 0.5*ax.get_ylim()[0]+0.5*ax.get_ylim()[1], ax.get_ylim()[1], label="raw_syb_start", color="lightgreen")
     ax.vlines(syb_dataset["uncorrected_end"], ax.get_ylim()[0], 0.5*ax.get_ylim()[0]+0.5*ax.get_ylim()[1], label="raw_syb_end", color="pink")
     for s,e, t in zip(syb_dataset["uncorrected_start"].to_numpy(), syb_dataset["uncorrected_end"].to_numpy(), syb_dataset["syb_name"].to_numpy()):
        ax.text(0.5*s + 0.5*e, 0.8*ax.get_ylim()[0]+0.2*ax.get_ylim()[1], t)
    #  ax.legend(loc='upper left')

def draw_filtered_song(ax: plt.Axes):
     song_dataset["filtered_song"].drop_vars("song_fs").plot.line(ax=ax, label="filtered_song", color="blue")
     song_dataset["amp"].drop_vars("song_fs").plot.line(ax=ax, label="filtered_amp", color="orange")
     ax.axhline(syb_dataset["amp_threshold"].item(), color="black", linestyle=":", label="amp_threshold")
     ax.vlines(syb_dataset["start"], (ax.get_ylim()[0]+ax.get_ylim()[1])/2, ax.get_ylim()[1], label="syb_start", color="green")
     ax.vlines(syb_dataset["end"], ax.get_ylim()[0], (ax.get_ylim()[0]+ax.get_ylim()[1])/2, label="syb_end", color="red")
     tmp = syb_dataset.where((syb_dataset["uncorrected_start"]!= syb_dataset["start"]) | (syb_dataset["uncorrected_end"]!= syb_dataset["end"]), drop=True)
     ax.vlines(tmp["uncorrected_start"], 0.2*ax.get_ylim()[0]+0.8*ax.get_ylim()[1], ax.get_ylim()[1], label="raw_syb_start", color="lightgreen")
     ax.vlines(tmp["uncorrected_end"], ax.get_ylim()[0], 0.8*ax.get_ylim()[0]+0.2*ax.get_ylim()[1], label="raw_syb_end", color="pink")
     ax.vlines(selected_subsyb["subsyllable_t"], ax.get_ylim()[0], ax.get_ylim()[1], color="black", linestyle=":", label=f'subsyllable{params["subsyllable_name"]}_t')
     ax.scatter(selected_subsyb["subsyllable_t"], selected_subsyb["accoustic_features"].sel(accoustic="amp"), label="amp", color="yellow", marker="+", zorder=2)

def draw_spectrogram(ax: plt.Axes):
     ax.specgram(song_dataset["filtered_song"].to_numpy(), Fs=song_dataset["song_fs"].item(), NFFT=512, xextent=(song_dataset["t"].min().item(), song_dataset["t"].max().item()))
     ax.vlines(selected_syb["start"], (ax.get_ylim()[0]+ax.get_ylim()[1])/2, ax.get_ylim()[1], label="syb_start", color="green")
     ax.vlines(selected_syb["end"], ax.get_ylim()[0], (ax.get_ylim()[0]+ax.get_ylim()[1])/2, label="syb_end", color="red")
     ax.vlines(selected_subsyb["subsyllable_t"], ax.get_ylim()[0], ax.get_ylim()[1], color="black", linestyle=":", label=f'subsyllable{params["subsyllable_name"]}_t')
     ax.scatter(selected_subsyb["subsyllable_t"], selected_subsyb["accoustic_features"].sel(accoustic="pitch"), label="pitch", color="orange", marker="+", zorder=2)
     ax.twinx().scatter(selected_subsyb["subsyllable_t"], selected_subsyb["accoustic_features"].sel(accoustic="entropy"), label="entropy", color="blue", marker="+", zorder=2)
     
def draw_neuro(ax: plt.Axes):
     tx = ax.twinx()
     for i, v in enumerate(selected_subsyb["neuro"].to_numpy()):
        
        if "_ifr" in v:
             m_ax = tx
             neuro_dataset["ifr"].sel(neuron=v[:-4]).plot.line(ax=m_ax, color=f"C{i}", label=v, alpha=0.5)
        elif "_bua" in v:
          m_ax = ax
          neuro_dataset["bua"].sel(sensor=v[:-4]).plot.line(ax=m_ax, color=f"C{i}", label=v, alpha=0.5)
        m_ax.scatter(selected_subsyb["subsyllable_t"], selected_subsyb["neuro_features"].sel(neuro=v, lag=params["lag"]), label=f'{v}_lag{params["lag"]*1000}ms', color=f"C{i}", marker="+", zorder=10)
        m_ax.quiver(selected_subsyb["subsyllable_t"] - params["lag"], selected_subsyb["neuro_features"].sel(neuro=v, lag=params["lag"]), params["lag"], 0, zorder=5, color=f"C{i}", angles="xy", scale=1, scale_units="xy", width=0.002)
     ax.vlines(selected_subsyb["subsyllable_t"], ax.get_ylim()[0], ax.get_ylim()[1], color="black", linestyle=":", label=f'subsyllable{params["subsyllable_name"]}_t')
     ax.set_title("")
     tx.set_title("")

def draw_accoustic_feat(ax: plt.Axes, k: int):
     feat = selected_subsyb["accoustic"].isel(accoustic=k).item()
     df = selected_subsyb["accoustic_features"].isel(accoustic=k).to_dataframe()
     df[feat] = df.pop("accoustic_features")
     sns.histplot(df[[feat]], ax=ax, bins=20, kde=True)

def draw_neuro_feat(ax: plt.Axes, k: int):
     feat = selected_subsyb["neuro"].isel(neuro=k).item()
     df = selected_subsyb["neuro_features"].isel(neuro=k).to_dataframe()
     df[feat] = df.pop("neuro_features")
     sns.histplot(df[[feat]], ax=ax, bins=20, kde=True)

def draw_acoustic2neuro_model_score(ax: plt.Axes, k: int):
     m= models_dataset.sel(lag = params["lag"], subsyllable_name=params["subsyllable_name"]).isel(neuro=k)
     bt_df = m["bootstrap_accoustic2neuro_score"].to_dataframe().reset_index()
     name = f'predicting_{models_dataset["neuro"].isel(neuro=k).item()}'
     bt_df[name] = bt_df["bootstrap_accoustic2neuro_score"]
     sns.histplot(bt_df[[name]], ax=ax, bins=20, kde=True)
     ax.axvline(m["accoustic2neuro_score"].item(), color="red")
     ax.text(m["accoustic2neuro_score"].item()+0.02*(ax.get_xlim()[1] - ax.get_xlim()[0]), 0.5*ax.get_ylim()[0] + 0.5*ax.get_ylim()[1], f'pvalue={m["accoustic2neuro_pvalue"].item()}', zorder=4)
     ax.set_title("")

def draw_neuro2accoustic_model_score(ax: plt.Axes, k: int):
     m= models_dataset.sel(lag = params["lag"], subsyllable_name=params["subsyllable_name"]).isel(accoustic=k)
     bt_df = m["bootstrap_neuro2accoustic_score"].to_dataframe().reset_index()
     name = f'predicting_{models_dataset["accoustic"].isel(accoustic=k).item()}'
     bt_df[name] = bt_df["bootstrap_neuro2accoustic_score"]
     sns.histplot(bt_df[[name]], ax=ax, bins=20, kde=True)
     ax.axvline(m["neuro2accoustic_score"].item(), color="red")
     ax.text(m["neuro2accoustic_score"].item()+0.02*(ax.get_xlim()[1] - ax.get_xlim()[0]), 0.5*ax.get_ylim()[0] + 0.5*ax.get_ylim()[1], f'pvalue={m["neuro2accoustic_pvalue"].item()}', zorder=4)
     ax.set_title("")

def draw_accoustic2neuro_recap(ax: plt.Axes, k: int, cbarax):
     models_dataset["subsyllable_name"] = models_dataset["subsyllable_name"].astype(str)
     np.log10(models_dataset["accoustic2neuro_pvalue"]).rename("log10(p_value)").isel(neuro=k).plot.pcolormesh(ax=ax, x="subsyllable_name", y="lag", vmin=-5, vmax=0,cbar_ax=cbarax)
     ax.set_title("")
     # sns.scatterplot(df1, x="subsyllable_name", y="lag", hue="accoustic2neuro_pvalue", ax=ax, legend=False, hue_norm=(0, 1), palette="viridis")

def draw_neuro2accoustic_recap(ax: plt.Axes, k: int, cbarax):
     models_dataset["subsyllable_name"] = models_dataset["subsyllable_name"].astype(str)
     np.log10(models_dataset["neuro2accoustic_pvalue"]).rename("log10(p_value)").isel(accoustic=k).plot.pcolormesh(ax=ax, x="subsyllable_name", y="lag", vmin=-5, vmax=0, cbar_ax=cbarax)
     ax.set_title("")
     # df1 = models_dataset["neuro2accoustic_pvalue"].isel(accoustic=k).to_dataframe().reset_index()
     # sns.scatterplot(df1, x="subsyllable_name", y="lag", hue="neuro2accoustic_pvalue", ax=ax, legend=False, hue_norm=(0, 1), palette="viridis")

f =  plt.figure(layout="tight")
f.suptitle(str(params))
i=0
grid = matplotlib.gridspec.GridSpec(10, 25, figure=f, hspace=0.3, wspace=0.2, left=0.05, right=0.85, top=0.9, bottom=0.1)
cbar = f.add_subplot(grid[8:,  24:25])
# legend_ax = f.add_subplot(grid[:, slice(10,12)])
song_ax: plt.Axes = f.add_subplot(grid[i, slice(0,24)])
i+=1
filtered_ax = f.add_subplot(grid[i, slice(0,24)])
song_ax.sharex(filtered_ax)
i+=1
spectrogram_ax = f.add_subplot(grid[i, slice(0,24)])
filtered_ax.sharex(spectrogram_ax)
i+=1
neuro_ax = f.add_subplot(grid[i, slice(0,24)])
spectrogram_ax.sharex(neuro_ax)

i+=1
accoustic_feature_axs = [f.add_subplot(grid[i, slice(k*int(24/n_acoustics),(k+1)*int(24/n_acoustics))]) for k in range(n_acoustics)]
i+=1
neuro_feature_axs = [f.add_subplot(grid[i, slice(k*int(24/n_neuro),(k+1)*int(24/n_neuro))]) for k in range(n_neuro)]


i+=1
neuro2accoustic_model_axs = [f.add_subplot(grid[i, slice(k*int(24/n_acoustics),(k+1)*int(24/n_acoustics))]) for k in range(n_acoustics)]
i+=1
accoustic2neuro_model_axs = [f.add_subplot(grid[i, slice(k*int(24/n_neuro),(k+1)*int(24/n_neuro))]) for k in range(n_neuro)]

i+=1
neuro2accoustic_recap_axs = [f.add_subplot(grid[i, slice(k*int(24/n_acoustics),(k+1)*int(24/n_acoustics))]) for k in range(n_acoustics)]
i+=1
accoustic2neuro_recap_axs = [f.add_subplot(grid[i, slice(k*int(24/n_neuro),(k+1)*int(24/n_neuro))]) for k in range(n_neuro)]


draw_raw_song(song_ax)
draw_filtered_song(filtered_ax)
draw_spectrogram(spectrogram_ax)
draw_neuro(neuro_ax)
for k, ax in enumerate(accoustic_feature_axs):
     draw_accoustic_feat(ax, k)
for k, ax in enumerate(neuro_feature_axs):
     draw_neuro_feat(ax, k)

for k, ax in enumerate(neuro2accoustic_model_axs):
     draw_neuro2accoustic_model_score(ax, k)
for k, ax in enumerate(accoustic2neuro_model_axs):
     draw_acoustic2neuro_model_score(ax, k)

for k, ax in enumerate(neuro2accoustic_recap_axs):
     draw_neuro2accoustic_recap(ax, k, cbar)
for k, ax in enumerate(accoustic2neuro_recap_axs):
     draw_accoustic2neuro_recap(ax, k, cbar)
f.legend(bbox_to_anchor=(0.85, 0, 0.15, 1), loc="center")
for ax in f.axes:
     ax.tick_params(labelsize=8, direction="in")
plt.show()
# spectrogram_ax: plt.Axes = f.add_subplot(grid[1, :])
# feats_ax: plt.Axes = f.add_subplot(grid[2, :])
# bua_ax: plt.Axes = f.add_subplot(grid[3, :])
# amp_ax_dist = f.add_subplot(grid[4, slice(0, 3)])
# entropy_ax_dist = f.add_subplot(grid[4, slice(3, 6)])
# pitch_ax_dist = f.add_subplot(grid[4, slice(6, 9)])
# bua_ax_dist = f.add_subplot(grid[4, slice(9, 12)])

# m_amp_ax_trs = f.add_subplot(grid[5, slice(0, 2)])
# m_entropy_ax_trs = f.add_subplot(grid[5, slice(2, 4)])
# m_pitch_ax_trs = f.add_subplot(grid[5, slice(4, 6)])
# ms_amp_ax_trs = f.add_subplot(grid[5, slice(6, 8)])
# ms_entropy_ax_trs = f.add_subplot(grid[5, slice(8, 10)])
# ms_pitch_ax_trs = f.add_subplot(grid[5, slice(10, 12)])
# m_bt_ax = f.add_subplot(grid[6, slice(0, 6)])
# amp_bt_ax = f.add_subplot(grid[6, slice(6, 8)])
# entropy_bt_ax = f.add_subplot(grid[6, slice(8, 10)])
# pitch_bt_ax = f.add_subplot(grid[6, slice(10, 12)])

# song_ax.sharex(spectrogram_ax)
# spectrogram_ax.sharex(feats_ax)
# feats_ax.sharex(bua_ax)

# amp_ax_dist.sharex(m_amp_ax_trs)
# # m_amp_ax_trs.sharex(amp_bt_ax)
# # m_entropy_ax_trs.sharex(entropy_bt_ax)
# entropy_ax_dist.sharex(m_entropy_ax_trs)
# pitch_ax_dist.sharex(m_pitch_ax_trs)
# # m_pitch_ax_trs.sharex(pitch_bt_ax)

# bua_ax_dist.sharex(ms_amp_ax_trs)
# ms_amp_ax_trs.sharex(ms_entropy_ax_trs)
# ms_entropy_ax_trs.sharex(ms_pitch_ax_trs)

# song.plot.line(ax=song_ax, zorder=1)
# amp.plot.line(ax=song_ax)
# song_ax.axhline(label_threshold, color="black", linestyle=":")


# song_ax.vlines(labels["start_index"]/song_fs, song.min().item(), song.max(), color="green", alpha=0.5)
# song_ax.vlines(labels["end_index"]/song_fs, song.min().item(), song.max(), color="red", alpha=0.5)
# # song_ax.vlines(labels["uncorrected_start_index"]/song_fs, song.min().item(), song.max(), color="lightgreen", linestyle=":")
# # song_ax.vlines(labels["uncorrected_end_index"]/song_fs, song.min().item(), song.max(), color="pink", linestyle=":")
# song_ax.scatter(labels2["subsybsongpos"].to_numpy()/song_fs,np.ones(labels2["subsybsongpos"].shape) * (song.max().item())/2, color="black")
# for _, row in labels.iterrows():
#     song_ax.text((row["uncorrected_start_index"]+row["uncorrected_end_index"])/(2*song_fs), song.max()*0.8, row["syb_name"])
# for _, row in labels2.iterrows():
#     spectrogram_ax.axvline(row["start_index"]/song_fs, color="green")
#     spectrogram_ax.axvline(row["end_index"]/song_fs, color="red")
#     # spectrogram_ax.axvline(row["uncorrected_start_index"]/song_fs, color="lightgreen", linestyle=":")
#     # spectrogram_ax.axvline(row["uncorrected_end_index"]/song_fs, color="pink", linestyle=":")
#     spectrogram_ax.axvline((row["subsybsongpos"] - int(song_fs/200))/song_fs, color="black", linestyle=":")
#     spectrogram_ax.axvline((row["subsybsongpos"] + int(song_fs/200))/song_fs, color="black", linestyle=":")

# spectrogram_ax.specgram(song, Fs=song_fs, zorder=1)

# spectrogram_ax.scatter(features["subsybsongpos"]/song_fs, features["pitch"], color="orange", label="pitch")
# feats_ax.scatter(features["subsybsongpos"]/song_fs, features["entropy"], color="orange", label="entropy")
# feats_ax.legend()
# song_ax.scatter(features["subsybsongpos"]/song_fs, features["amp"], color="orange", zorder=2, label="amplitude")
# sns.histplot(features["amp"], ax=amp_ax_dist, bins=20, kde=True)
# sns.histplot(features["entropy"], ax=entropy_ax_dist, bins=20, kde=True)
# sns.histplot(features["pitch"], ax=pitch_ax_dist, bins=20, kde=True)
# sns.histplot(syb_bua[f"bua_lag{lag}"], ax=bua_ax_dist, bins=20, kde=True)

# bua.plot.line(ax=bua_ax)
# bua_ax.scatter(syb_bua["subsybsongpos"]/song_fs, syb_bua[f"bua_lag{lag}"], color="red", label=f"lag={lag}ms")
# bua_ax.scatter(syb_bua["subsybsongpos"]/song_fs - lag*10**(-3), syb_bua[f"bua_lag{lag}"], color="red", marker="+")
# bua_ax.legend()


# m_amp_ax_trs.scatter(features["amp"], transformed, color="red", marker="+", label="transformed")
# m_amp_ax_trs.scatter(features["amp"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+", label="real")
# m_amp_ax_trs.legend()

# m_entropy_ax_trs.scatter(features["entropy"], transformed, color="red", marker="+")
# m_entropy_ax_trs.scatter(features["entropy"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+")

# m_pitch_ax_trs.scatter(features["pitch"], transformed, color="red", marker="+")
# m_pitch_ax_trs.scatter(features["pitch"], syb_bua[f"bua_lag{lag}"], color="orange", marker="+")

# sns.histplot(bt, x="score", ax=m_bt_ax, bins=20)
# m_bt_ax.axvline(score, color="red")
# m_bt_ax.text(score*1.05, m_bt_ax.get_ylim()[1]/2, f"p-value={p_value}", color="red")

# ms_entropy_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["entropy", "transformed"], color="red", marker="+")
# ms_entropy_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["entropy"], color="orange", marker="+")

# ms_amp_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["amp", "transformed"], color="red", marker="+")
# ms_amp_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["amp"], color="orange", marker="+")

# ms_pitch_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], feat_models.loc["pitch", "transformed"], color="red", marker="+")
# ms_pitch_ax_trs.scatter(syb_bua[f"bua_lag{lag}"], features["pitch"], color="orange", marker="+")

# for feat, ax in {"amp": amp_bt_ax, "entropy": entropy_bt_ax, "pitch": pitch_bt_ax}.items():
#     sns.histplot(pd.DataFrame(feat_models.loc[feat, "bt_score_dist"].reshape(-1, 1), columns=["score"]), x="score", ax=ax, bins=20)
#     ax.axvline(feat_models.loc[feat, "score"], color="red")
#     ax.text(feat_models.loc[feat, "score"]*1.05, ax.get_ylim()[1]/2, f'p-value={feat_models.loc[feat, "p_value"]}', color="red")

# plt.show()