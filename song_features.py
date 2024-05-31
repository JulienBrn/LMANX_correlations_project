from pathlib import Path
from regular_array import RegularArray, regsel
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import logging, beautifullogger, tqdm.auto as tqdm
import pickle
import scipy.signal
from common import load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base
logger = logging.getLogger(__name__)
beautifullogger.setup()
xr.set_options(display_expand_coords=True, display_max_rows=100)

if not __name__ == "__main__":
    exit()


# ============================================================================ #
# Functions
# ============================================================================ #

def get_pitch(a, freqs):
    f_fs = 1
    a = np.apply_along_axis(lambda arr: np.interp(np.arange(int(freqs[0]), int(freqs[-1]*f_fs))/f_fs, freqs, arr), -1, a)
    freqs = np.fft.rfftfreq(a.shape[-1], f_fs)
    pitches = 1/freqs
    r = np.searchsorted(pitches, 200)
    fft = np.abs(np.fft.rfft(a, axis=-1))
    max = 1+np.argmax(fft[(slice(None),) * (fft.ndim-1)+(slice(1, r),)], axis=-1)
    return pitches[max]

def get_pitchv2(candidate, vals, freqs):
    powers = np.arange(0, np.log2(freqs[-1] / candidate)+1)
    f = candidate * np.power(2, powers)
    v = np.interp(f, freqs, vals)
    res = (np.sum(v)/powers.size) * (powers.size**(3/4))
    return res

def computation(d: xr.Dataset, s: Path):
    # d=d.drop_dims([dim for dim in d.dims if not dim in ["t_song", "t_bin", "f"]])
    t_bin_fs = d["t_bin_coords"].data.fs
    d["t_song"] = d["t_song_coords"].data.arr
    song_fs = d["t_song_coords"].data.fs
    amp_window = d["filtered_song"].rolling(t_song=int(song_fs/500), center=True).construct(window_dim="t_win", stride=int(song_fs/t_bin_fs)).rename(t_song="t_bin_w")
    amp = np.abs(amp_window).mean(dim="t_win", skipna=False)
    d["amp"] = amp.interp(t_bin_w = d["t_bin_coords"].data.arr).rename(t_bin_w="t_bin")
    pdf = d["spectrogram"]/d["spectrogram"].sum("f")
    d["entropy"] = (-pdf*np.log(pdf)).sum("f")
    d["pitch"] = xr.apply_ufunc(get_pitch, np.log(d["spectrogram"]), d["f"], input_core_dims=[["f"]]*2)
    # d["candidate"] = xr.DataArray(np.arange(100, 8000), dims="candidate")
    # d["candidate_value"] = xr.apply_ufunc(get_pitchv2, d["candidate"], d["spectrogram"], d["f"], input_core_dims=[[]] + [["f"]]*2, vectorize=True)
    # d["pitch2"] = d["candidate_value"].argmax("candidate")
    f, axs=plt.subplots(3, sharex=True)
    d["pitch"].plot.line(ax=axs[0], color="red", label="pitch")
    np.log(d["spectrogram"]).plot.pcolormesh(ax=axs[0])
    d["entropy"].plot.line(ax=axs[1], color="blue", label="entropy")
    d["amp"].plot.line(ax=axs[1].twinx(), color="red", label="amp")
    d["pitch"].plot.line(ax=axs[0], color="red", label="pitch")
    d["pitch"].plot.line(ax=axs[2], color="red", label="pitch")
    # d["pitch2"].plot.line(ax=axs[0], color="blue", label="pitch2")
    axs[1].legend()
    plt.show()
    # print(d)
    # exit()
    return d


# ============================================================================ #
# Run
# ============================================================================ #

stop_on_error=False
pd.Series.buffered_errors=not stop_on_error
write_output=False


def requires(s):
    return [get_session_storage_path(s)/"song/song_filtered.pkl", get_session_storage_path(s)/"song/song_filtered_spectrogram.pkl"]

def generates(s) -> Path:
    return get_session_storage_path(s)/"song/song_features.pkl"

sessions = retrieve_sessions(requires, generates, recompute_existing=True, n_per_subject=3)

errors = []

for s in tqdm.tqdm(sessions):
    try:
        if len(requires(s)) > 0:
            d = xr.merge([load_pickle(f) for f in requires(s)])
        else:
            d= xr.Dataset()

        res = computation(d, s)
        if write_output:
            out = generates(s)
            out.parent.mkdir(exist_ok=True, parents=True)
            tmp_file = out.parent/ (out.stem + ".tmp"+out.suffix)
            with  tmp_file.open("wb") as f:
                pickle.dump(res, f)
            tmp_file.rename(out)
        else:
            print(res)
    except Exception as e:
        e.add_note(f'During computation for session {s}')
        errors.append(e)
        if stop_on_error:
            raise
        else:
            logger.error(f'Error during computation for session {s}')

if len(errors) > 0:
    if len(errors) > 1:
        raise ExceptionGroup(f"Found {len(errors)} errors", errors)
    else:
        raise errors[0]

