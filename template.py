from pathlib import Path
from regular_array import RegularArray
import numpy as np, pandas as pd, xarray as xr
import logging, beautifullogger, tqdm.auto as tqdm
import pickle, yaml
from common import load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base
logger = logging.getLogger(__name__)
beautifullogger.setup()
xr.set_options(display_expand_coords=True, display_max_rows=100)

if not __name__ == "__main__":
    exit()

# ============================================================================ #
# Functions
# ============================================================================ #

def computation(d: xr.Dataset, s: Path): pass

# ============================================================================ #
# Run
# ============================================================================ #

stop_on_error=True
pd.Series.buffered_errors=not stop_on_error
write_output=False

def requires(s):
    return [get_session_source_path(s)/"metadata.pkl"]

def generates(s) -> Path:
    return get_session_storage_path(s)/"song/stuff.pkl"

sessions = retrieve_sessions(requires, generates, ignore_existing=False)

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

