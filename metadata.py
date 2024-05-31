from pathlib import Path
from regular_array import RegularArray
import numpy as np, pandas as pd, xarray as xr
import logging, beautifullogger, tqdm.auto as tqdm
import pickle
from common import load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base, retrieve_tasks
from common import singleglob

logger = logging.getLogger(__name__)
beautifullogger.setup()
xr.set_options(display_expand_coords=True, display_max_rows=100)

if not __name__ == "__main__":
    exit()



def get_smrx_candidates(s: Path, only_ok=False):
    return singleglob(get_session_source_path(s), "*.smr", "*.smrx", "*.SMR", "*.SMRX", only_ok=only_ok)

# ============================================================================ #
# SessionPart
# ============================================================================ #



def get_session_metadata(s: Path):
    subject = s.parents[-2].stem
    metadata = dict(
        carmen_session_folder = str(s),
        session_name = s.stem,
        subject = subject,
        session_unique_id = subject[0] + subject[-2:] + "---"+ s.stem,
        date = "",
        experimenter="",
        source_data_type="smrx" if get_smrx_candidates(s, only_ok=True) else
                         "npx" if (get_session_source_path(s)/"bua"/ "raw_traces").exists() else "unknown"
        
    )
    # series = pd.Series(metadata)
    # series.index.name = "session_metadata_name"
    # print(series)
    # res = xr.DataArray.from_series(series)
    return metadata
    

# ============================================================================ #
# RawElectrophyPart
# ============================================================================ #

invalidate_smrx = False

def get_brain_structure(subject, session_name, channel_name):
    rules = dict(
        B60 = lambda n: "RA" if n=="CSC16" else "X" if n=="CSC17" else "LMAN" if n=="CSC18" else "",
        Bird47 = lambda n: "X" if n=="CSC7" else "LMAN" if n=="CSC8" else "",
        Green23 = lambda n: ("" if not n in ["CSC1", "CSC2", "CSC3", "CSC4"] else
            "X" if session_name >= "2015-03-10" and session_name <= "2015-03-13" and "2015_03_12_10_04_26" not in session_name else "LMAN"),
        GreenGreen = lambda n: ("" if not n in ["CSC2", "CSC3", "CSC4"] else
            "X" if session_name in ["1_29_15_14_45_35", "1_29_15_17_25_31"] else "LMAN"),
        Red2 = lambda n: ("" if not n in ["SPK 002", "SPK 003", "SPK 004"] else
            "X" if session_name >= "red2_0036" and session_name <= "red2_0040" else "LMAN"),
    )
    return rules[subject](channel_name)

def get_raw_electrophy_coordinates(s: Path, meta: dict):
    res = xr.Dataset()
    match meta["source_data_type"]:
        case "smrx":
            import subprocess
            smrx_file = get_smrx_candidates(s)
            metadata_file = get_session_storage_path(s)/"electrophy_metadata.tmp.tsv"
            if not smrx_file.exists():
                raise Exception(f"Could not find smrx file for session {meta['carmen_session_folder']} expected at {str(smrx_file.relative_to(source_base))}")
            if not metadata_file.exists() or invalidate_smrx:
                cmd = ['conda', 'run', '-n', 'spike2', 'smrx2python', "--channels_only",
                            '-i', str(smrx_file), 
                            '-o', str(metadata_file),
                            '--noprogress',
                            '-logginglevel', str(logging.WARNING)
                ]
                subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if not metadata_file.exists():
                    raise Exception("Problem")
            channel_metadata = pd.read_csv(metadata_file, sep="\t", index_col=False)
            channel_metadata["valid"] = channel_metadata["name"].apply(lambda n: (get_session_source_path(s)/"bua"/(n+"_envelope_1000Hz.npy")).exists()).astype(np.uint8)
            if channel_metadata.loc[channel_metadata["valid"]].groupby(["fs", "n_data"]).ngroups !=1:
                raise Exception(f'Expected a single fs for all electrophy channels got:\n{channel_metadata.loc[channel_metadata["valid"]]}')
            fs = channel_metadata["fs"].iat[0]
            n_data = int(channel_metadata["n_data"].iat[0])
            start=0
            
            channel_metadata = channel_metadata.loc[(channel_metadata["fs"] == fs) & (channel_metadata["n_data"].astype(int) == n_data)]
            channel_metadata["BrainStructure"] = channel_metadata["name"].apply(lambda n: get_brain_structure(meta["subject"], meta["session_name"], n))

            channel_metadata["sensor"] = channel_metadata.pop("name")
            channel_metadata = channel_metadata.drop(columns=["smrx_type", "n_data", "data_kind", "fs"])
            
        case "npx":
            import json
            with (source_base / s / "metadata.json").open("r") as f:
                d = json.load(f)
            with (source_base / s / "synchro_imecap_corr_on_nidq.json").open("r") as f:
                sync = json.load(f)
            channel_metadata = pd.Series(d["channel_location"], name="BrainStructure")
            channel_metadata.index.name = "sensor"
            channel_metadata = channel_metadata.reset_index()
            channel_metadata["valid"] = True
            channel_metadata["n_data"] = channel_metadata["sensor"].apply(lambda n: np.load(source_base / s / "bua"/ "raw_traces"/(n+".npy"), mmap_mode='r').size)
            n_data = channel_metadata["n_data"].iat[0]

            if not (channel_metadata["n_data"] == n_data).all():
                channel_metadata["max_time"] = channel_metadata["n_data"] / d["fs_ap"]
                raise Exception(f'n_data problem\n{channel_metadata}')
            
            fs = d["fs_ap"] * (1/sync["a"])
            start = sync["b"]

            channel_metadata = channel_metadata.drop(columns=["n_data"])
            
        case _:
            raise Exception("Unhandled source type")
    
    res["t_raw_coords"] = xr.DataArray(RegularArray.from_params(start=start, fs=fs, ndata=n_data), dims="t_raw")
    df = channel_metadata.set_index("sensor")
    df = df.stack()
    df.index.names = ["sensor", "sensor_metadata"]
    res["sensor_metadata_values"] = xr.DataArray.from_series(df)
    res["valid_sensor"] = res["sensor_metadata_values"].sel(sensor_metadata="valid")
    return res

# ============================================================================ #
# UnitElectrophyPart
# ============================================================================ #

def get_unit_electrophy_coordinates(s: Path, meta: dict):
    res = xr.Dataset()
    import json
    with (get_session_source_path(s) / "metadata.json").open("r") as f:
        d = json.load(f)
    units_location_mapping = pd.DataFrame()
    units_location_mapping["unit_name"] = d["units_analysis"].keys()
    units_location_mapping["unit_sensor"] = d["units_analysis"].values()
    units_location_mapping["carmen_unit_path"] = units_location_mapping["unit_name"].buffered_apply(
        lambda n: str(singleglob(get_session_source_path(s), f"units/**/{n}.txt", f"unused/**/{n}.txt").relative_to(get_session_source_path(s)))
    )
    units_location_mapping["valid"] = units_location_mapping["unit_name"].apply(
         lambda n: singleglob(get_session_source_path(s), f"units/**/{n}.txt", only_ok=True))
    missing_errors=[]
    for p in get_session_source_path(s).glob("units/**/*.txt"):
        if not str(p.relative_to(get_session_source_path(s))) in units_location_mapping["carmen_unit_path"].to_list():
            try:
                raise Exception(f"Unit {p.relative_to(get_session_source_path(s))} expected in metadata but not found")
            except Exception as e:
                missing_errors.append(e)
    match len(missing_errors):
        case 0:pass
        case 1:
            raise missing_errors[0]
        case _:
            raise ExceptionGroup("Several units are missing in metadata", missing_errors)
    
    units_location_mapping["n_spikes"] = units_location_mapping["carmen_unit_path"].apply(lambda n: np.loadtxt(get_session_source_path(s) / n).size)
    df = units_location_mapping.set_index("unit_name")
    df = df.stack()
    df.index.names = ["unit", "unit_metadata"]
    res["unit_metadata_values"] = xr.DataArray.from_series(df)
    res["spike_unit"] = xr.DataArray(np.repeat(units_location_mapping["unit_name"].to_numpy(), units_location_mapping["n_spikes"].to_numpy()), dims="spike_id")
    res["spike_index"] = xr.DataArray(np.concatenate([np.arange(n) for n in units_location_mapping["n_spikes"]]), dims="spike_id")
    res["valid_unit"] = res["unit_metadata_values"].sel(unit_metadata="valid")
    return res

# ============================================================================ #
# SongPart
# ============================================================================ #

def get_song_coordinates(s: Path, meta: dict):
    res = xr.Dataset()
    import json
    with (get_session_source_path(s) / "metadata.json").open("r") as f:
        d = json.load(f)
    song_fs = d["fs_mic"]
    song_channel = d["song_channel"]
    song_file = (get_session_source_path(s) / "song" / f"{song_channel}.npy")
    n_data = np.load(song_file, mmap_mode="r").size
    res["t_song_coords"] = xr.DataArray(RegularArray.from_params(start=0, fs=song_fs, ndata=n_data), dims="t_song")
    
    label_files = list((get_session_source_path(s) / "song").glob("*labels.txt"))
    if len(label_files) != 1:
        raise Exception(f'found {len(label_files)} label files')
    label_file = label_files[0]

    labels_csv = pd.read_csv(label_file, sep=",", index_col=False, names=["start_index", "end_index", "syb"])
    
    res["syb_num"] = xr.DataArray(np.arange(len(labels_csv.index)), dims="syb_num")

    song_metadata = dict(
        carmen_song_file = str(song_file.relative_to(get_session_source_path(s))),
        carmen_label_file= str(label_file.relative_to(get_session_source_path(s))),
        song_channel = song_channel,
    )
    res["song_metadata"] = xr.DataArray(list(song_metadata.keys()), dims="song_metadata")
    res["song_metadata_values"] = xr.DataArray(list(song_metadata.values()), dims="song_metadata")

    motifs = dict(
        B60 = ["a,b,c,d", "a,z,d", "a,c,d", "a,b"],
        Bird47 = ["a,b,c,e", "a,b,d", "a,b"],
        BlackCyan = ["a,b,c,e", "a,b,d", "a,b"],
        Green23 = ["a,b", "a,b,c,d"],
        GreenGreen = ["a,b", "a,b,c,d"],
        Red2 = ["a,b", "a,b,c,d"],
        RedRed = ["a,b1,b2,b3"],
    )

    res["motifs"] = xr.DataArray(motifs[meta["subject"]], dims="motif_index").astype(str).astype(object)

    return res
    



# ============================================================================ #
# Functions
# ============================================================================ #

def computation(d: xr.Dataset, s: Path): 
    session_metadata = get_session_metadata(s)

    tmp = pd.Series(session_metadata)
    tmp.index.name = "session_metadata"
    d["session_metadata_values"] = xr.DataArray.from_series(tmp)

    raw_electrophy_coordinates = get_raw_electrophy_coordinates(s, session_metadata)
    d = xr.merge([d, raw_electrophy_coordinates])

    unit_electrophy_coordinates = get_unit_electrophy_coordinates(s, session_metadata)
    d = xr.merge([d, unit_electrophy_coordinates])

    song_coordinates = get_song_coordinates(s, session_metadata)
    d = xr.merge([d, song_coordinates])
    d= d.set_coords(d.data_vars)
    return d

# ============================================================================ #
# Run
# ============================================================================ #

stop_on_error=False
pd.Series.buffered_errors=not stop_on_error
write_output=True


def requires():
    return []

def generates(session) -> Path:
    return get_session_storage_path(session)/"metadata.pkl"

sessions = retrieve_tasks(requires, generates, recompute_existing=True, n_per_group=None)
# exit()
errors = []

for s,  in tqdm.tqdm(sessions):
    try:
        if len(requires()) > 0:
            d = xr.merge([load_pickle(f) for f in requires()])
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
    raise ExceptionGroup(f"Found {len(errors)} errors", errors)

