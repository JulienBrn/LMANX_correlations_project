from pathlib import Path
from regular_array import RegularArray
import numpy as np, pandas as pd, xarray as xr
import logging, beautifullogger, tqdm.auto as tqdm
import pickle, yaml
from common import load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base, get_param
logger = logging.getLogger(__name__)
beautifullogger.setup()
xr.set_options(display_expand_coords=True, display_max_rows=100)

if not __name__ == "__main__":
    exit()


# ============================================================================ #
# Functions
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

def computation(d: xr.Dataset, s: Path):
    print(d)
    d=d.drop_dims([dim for dim in d.dims if not dim in ["t_song", "song_metadata"]])
    d["song_data"] = xr.DataArray(np.load(get_session_source_path(s)/d["song_metadata_values"].sel(song_metadata="carmen_song_file").item()).reshape(-1), dims="t_song")
    d["filtered_song"] = xr.apply_ufunc(filter_song, d["song_data"], d["t_song_coords"].data.fs, input_core_dims=[["t_song"], []], output_core_dims=[["t_song"]], vectorize=True)
    d=d.drop_dims("song_metadata")
    d = d[["filtered_song"]]
    return d


# ============================================================================ #
# Run
# ============================================================================ #

stop_on_error=True
pd.Series.buffered_errors=not stop_on_error
write_output=True


def requires(s):
    return [get_session_storage_path(s)/"metadata.pkl"]

def generates(s) -> Path:
    return get_session_storage_path(s)/"song/song_filtered.pkl"

sessions = retrieve_sessions(requires, generates, recompute_existing=True, n_per_subject=2)

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
            print(d["song_metadata_values"])
            raise
        else:
            logger.error(f'Error during computation for session {s}')

if len(errors) > 0:
    if len(errors) > 1:
        raise ExceptionGroup(f"Found {len(errors)} errors", errors)
    else:
        raise errors[0]

