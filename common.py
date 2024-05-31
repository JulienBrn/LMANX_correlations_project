# from template_helper import 
# load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import logging, tqdm.auto as tqdm
import pickle, yaml
import functools, inspect

logger = logging.getLogger(__name__)

source_base = Path("/media/filer2/T4b/Carmen/LMANX_correlations_project/LMANX_behavior_data/BirdData/")
# target_base = Path("~/Documents/Data/Carmen/LMANX_correlations_project")
target_base = Path("/media/filer2/T4b/Julien/Data/Carmen/LMANX_correlations_project")

@functools.cache
def get_params_dict():
    with Path("./analysis_params.yaml").open("r") as f:
        d = yaml.safe_load(f)
    return d

def get_param(p: str):
    return get_params_dict()[p]


def get_session_source_path(s: Path):
    return source_base / s
def get_session_storage_path(s: Path):
    return target_base/ "{subject}/{session_name}".format(subject=s.parents[-2].stem, session_name=s.stem)

def get_all_sessions(reload=False, progress=True, save=True, n_per_subject=None):
    saved_path = Path("./data/all_sessions.pkl")
    if not reload and saved_path.exists():
        with saved_path.open("rb") as f:
            all = pickle.load(f)
    else:
        all = [p.relative_to(source_base).parent for subject in tqdm.tqdm(list(source_base.iterdir()), disable= not progress) for p in subject.glob("**/Sessions/**/song")]
        if save:
            with saved_path.open("wb") as f:
                pickle.dump(all,f)
    subjects = [s.parents[-2].stem for s  in all]
    df = pd.DataFrame()
    df["path"] = all
    df["subject"] = subjects
    if not n_per_subject is None:
        df = df.groupby("subject").head(n_per_subject)
    return df["path"].to_list()
    
def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)
    
def save_result_decorator(func):
    saved_path = Path(f"./data/{func.__name__}.pkl")
    def new_f(*args, **kwargs):
        if saved_path.exists():
            with saved_path.open("rb") as f:
                res = pickle.load(f)
        else:
            res = func(*args, **kwargs)
            with saved_path.open("wb") as f:
                pickle.dump(res,f)
        return res
    return new_f



@save_result_decorator
def all_sessions(progress=True):
    res=pd.DataFrame()
    files = [p.relative_to(source_base).parent for subject in tqdm.tqdm(list(source_base.iterdir()), disable= not progress) for p in subject.glob("**/Sessions/**/song")]
    res["session"] = files
    res["subject"] = [s.parents[-2].stem for s  in files]
    return res

@save_result_decorator
def all_chunks(progress=True):
    sessions = all_sessions(progress)
    def get_chunks(row):
        if not (get_session_storage_path(Path(row["session"])) / "metadata.pkl").exists():
            return pd.Series()
        else:
            metadata = load_pickle(get_session_storage_path(Path(row["session"])) / "metadata.pkl")
            coords = metadata["t_song_coords"].data
            n_chunks = int(np.ceil((coords.end - coords.start)/get_param("spectrogram_chunk_size_seconds")))
            return pd.Series(np.arange(n_chunks))
    tqdm.tqdm.pandas(desc="getting n chunks")
    all_chunks: pd.DataFrame = sessions.progress_apply(get_chunks, axis=1).stack()
    all_chunks.index.names=["session_index", "chunk"]
    all_chunks = all_chunks.reset_index()[["session_index", "chunk"]]
    all_chunks = all_chunks.merge(sessions, how="outer", left_on="session_index", right_index=True)
    all_chunks = all_chunks.drop(columns=["session_index"])
    kept = all_chunks.loc[~pd.isna(all_chunks["chunk"])]
    if len(kept.index) != len(all_chunks.index):
        logger.warning(f"Removing some sessions because some metadata has not been computed...")
    return kept
        
            
def retrieve_tasks(requires, generates, recompute_existing=False, n_per_group=None, groups=["subject"], progress=True):
    gen_keys = list(inspect.signature(generates).parameters.keys())
    if "chunk" in gen_keys:
        all_tasks: pd.DataFrame = all_chunks(progress=progress)
    else:
        all_tasks = all_sessions(progress=progress)
    req_keys = list(inspect.signature(requires).parameters.keys())
    def is_possible(row: pd.Series):
        f = requires(*[row[k] for k in req_keys])
        return np.all([r.exists() for r in f])
    possible_tasks: pd.DataFrame = all_tasks.loc[all_tasks.apply(is_possible, axis=1)]
    n_discarded = len(all_tasks.index) - len(possible_tasks.index)
    if n_per_group is not None:
        grouped_tasks = possible_tasks.groupby(groups).head(n_per_group)
    else: 
        grouped_tasks = possible_tasks
    def is_already_done(row: pd.Series):
        f = generates(*[row[k] for k in gen_keys])
        return f.exists()
    already_done = grouped_tasks.apply(is_already_done, axis=1)
    n_already_done = already_done.sum()
    left = (~already_done).sum()
    if recompute_existing:
        logger.info(f"Got {len(all_tasks.index)} tasks. n_discarded = {n_discarded}, selected={len(grouped_tasks.index)}, n_already_done_but_recomputing = {n_already_done}, left = {left}.")
        res= grouped_tasks
    else:
        logger.info(f"Got {len(all_tasks.index)} tasks. n_discarded = {n_discarded}, selected={len(grouped_tasks.index)}, n_already_done = {n_already_done}, left = {left}.")
        res= grouped_tasks.loc[~already_done]
    return [tuple(x) for x in res[gen_keys].to_numpy()]



def retrieve_sessions(requires, generates, recompute_existing=False, n_per_subject=None, reload=False, progress=True, save=True):
    all_sessions = get_all_sessions(reload=reload, progress=progress, save=save, n_per_subject=n_per_subject)
    possible_sessions = [s for s in all_sessions if np.all([r.exists() for r in requires(s)])]
    n_discarded = len(all_sessions) - len(possible_sessions)
    sessions = [s for s in possible_sessions if not generates(s).exists()]

    n_already_done = len(possible_sessions) - len(sessions)
    if recompute_existing:
        logger.info(f"Got {len(all_sessions)} sessions. n_discarded = {n_discarded}, n_already_done_but_recomputing = {n_already_done}, left = {len(sessions)}.")
        return possible_sessions
    else:
        logger.info(f"Got {len(all_sessions)} sessions. n_discarded = {n_discarded}, n_already_done = {n_already_done}, left = {len(sessions)}.")
        return sessions

# def retrieve_tasks(requires, generates, recompute_existing=False, n_per_subject=None, reload=False, progress=True, save=True, keys={}):

    
def buffered_apply(self, f, *args, buffered=None, **kwargs):
    if buffered is None:
        buffered = self.buffered_errors
    if buffered:
        errors = []
        def new_f(*a, **kw):
            try:
                return f(*a, **kw)
            except Exception as e:
                errors.append(e)
                return pd.NA
        res = self.apply(new_f, *args, **kwargs)
        match len(errors):
            case 0:
                return res
            case 1:
                raise errors[0]
            case _:
                raise ExceptionGroup(f'Errors during computation of {f.__name__}', errors)
        
    else:
        return self.apply(f, *args, **kwargs)
    
pd.Series.buffered_apply= buffered_apply
pd.Series.buffered_errors=True




def singleglob(p: Path, *patterns, error_string='Found {n} candidates for pattern {patterns} in folder {p}', only_ok=False):
    all = [path for pat in patterns for path in p.glob(pat)]
    if only_ok:
        return len(all)==1
    if len(all) >1:
        raise Exception(error_string.format(p=p, n=len(all), patterns=patterns))
    if len(all) ==0:
        raise Exception(error_string.format(p=p, n=len(all), patterns=patterns))
    return all[0]


