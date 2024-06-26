from pathlib import Path
from regular_array import RegularArray, regsel
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import logging, beautifullogger, tqdm.auto as tqdm
import pickle
import scipy.signal
from common import load_pickle, retrieve_sessions, get_session_source_path, get_session_storage_path, source_base, target_base, get_param, retrieve_tasks
logger = logging.getLogger(__name__)
beautifullogger.setup()
xr.set_options(display_expand_coords=True, display_max_rows=100)

# if not __name__ == "__main__":
#     exit()


# ============================================================================ #
# Functions
# ============================================================================ #

# def compute_multispectrogram(sig: np.ndarray, song_t):
#     fs = song_t.fs
#     output_times = RegularArray.from_params(start=song_t.start, end= song_t.end, fs=get_param("t_bin_fs"), border_right="~=", border_left="~=")
#     window_size_start = int(get_param("t_spectrogram_min_window_seconds")*fs)
#     n_windows = int(1/(get_param("f_resolution_at_min_f") * get_param("t_spectrogram_min_window_seconds")))
#     sizes = [int(get_param("t_spectrogram_min_window_seconds")*fs*r) for r in np.linspace(1, 2, n_windows, endpoint=False)]
#     approx_n_freq_values = (get_param("max_song_f")-get_param("min_song_f") + 2*get_param("f_spectrogram_shoulders"))/ get_param("f_resolution_at_min_f")
#     # logger.debug(f"Total size of expected multispectrogram is ~{approx_n_freq_values} * {output_times.npoints} ~= {np.round((approx_n_freq_values * output_times.npoints)/(10**9), 2)}*10^9")
#     all_spectrograms = []
#     all_freqs=[]
#     for size in tqdm.tqdm(sizes, desc="Spectrograms", disable=True):
#         stft_obj = scipy.signal.ShortTimeFFT(scipy.signal.windows.hamming(size), hop=int(fs/get_param("t_bin_fs")), fs=fs, scale_to="psd")
#         _, p0 = stft_obj.lower_border_end
#         _, p1 = stft_obj.upper_border_begin(sig.size)
#         times = stft_obj.t(sig.size, p0, p1) + song_t.start
#         freqs = stft_obj.f
#         start_freq_index, end_freq_index = np.searchsorted(freqs, [get_param("min_song_f")-get_param("f_spectrogram_shoulders"), get_param("max_song_f")+get_param("f_spectrogram_shoulders")] )
#         spectrogram = stft_obj.spectrogram(sig, p0=p0, p1=p1)
#         spectrogram = spectrogram[start_freq_index:end_freq_index, :]
#         # print(spectrogram.nbytes)
#         freqs= freqs[start_freq_index:end_freq_index]
#         result_spectrogram = np.apply_along_axis(lambda a: np.interp(output_times.arr, times, a, left=np.nan, right=np.nan), axis=-1, arr= spectrogram)
#         all_spectrograms.append(result_spectrogram)
#         all_freqs.append(freqs)
#     return output_times, np.concatenate(all_freqs), np.concatenate(all_spectrograms, axis=-2)


# def as_xarray_spectrogram(times, freqs, spec):
#     arr = xr.DataArray(spec, dims=["f", "t_bin"])
#     # arr["t_bin"] = xr.DataArray(times.arr, dims="t_bin")
#     arr["f"] = xr.DataArray(freqs, dims="f")
#     arr = arr.sortby("f")
#     arr["t_bin_coords"] = xr.DataArray(times, dims="t_bin")
    
#     return arr
# mn = 0
def computation(d: xr.Dataset, s: Path):
    # global mn
    # mn+=1
    # if mn==1:
    #     return None
    # from HDShortFFT import HDShortFFT
    import scipy.signal
    d=d.drop_dims([dim for dim in d.dims if not dim in ["t_song"]])
    # start = d["t_song_coords"].data.start
    
    # chunk_start = start+ c*get_param("spectrogram_chunk_size_seconds") - get_param("t_spectrogram_min_window_seconds")
    # chunk_end = start+ (c+1)*get_param("spectrogram_chunk_size_seconds") + get_param("t_spectrogram_min_window_seconds")
    # print(int(1/(get_param("f_resolution_at_min_f") * get_param("t_spectrogram_min_window_seconds"))))
    # chunk_start = 17
    # chunk_end=23

    # chunk = regsel(d, t_song = slice(chunk_start,chunk_end))
    chunk = d
    fs = chunk["t_song_coords"].data.fs
    
    stft_obj = scipy.signal.ShortTimeFFT(scipy.signal.windows.hamming(int(get_param("t_spectrogram_min_window_seconds")*fs)), 
                          hop=int(fs/get_param("t_bin_fs")), 
                          fs=fs, 
                          scale_to="psd"
                        #   n_spectrogram=int(1/(get_param("f_resolution_at_min_f") * get_param("t_spectrogram_min_window_seconds")))
                        #    n_spectrogram=20
    )
    arr_size, p0, p1 = chunk["t_song_coords"].data.npoints, stft_obj.lower_border_end[1], stft_obj.upper_border_begin(chunk["t_song_coords"].data.npoints)[1]
    chunk["song_spectrogram"] = xr.apply_ufunc(lambda a: stft_obj.spectrogram(a, p0=p0, p1=p1, axis=-1), chunk["filtered_song"], input_core_dims=[["t_song"]], output_core_dims=[["f", "t_bin"]])
    chunk["t_bin"] = stft_obj.t(arr_size, p0, p1) + chunk["t_song_coords"].data.start
    chunk["t_bin_coords"] = xr.apply_ufunc(lambda a: RegularArray.from_array(a), chunk["t_bin"], input_core_dims=[["t_bin"]], output_core_dims=[["t_bin"]])
    chunk["f"] = stft_obj.f
    chunk = chunk.sel(f=slice(get_param("min_song_f")-get_param("f_spectrogram_shoulders"), get_param("max_song_f")+get_param("f_spectrogram_shoulders")))
    # print(chunk)
    # print(chunk["f"].to_series().to_string())
    # (chunk["f"] - chunk["f"].shift(f=1)).plot.line()
    # np.log(chunk["song_spectrogram"]).plot.pcolormesh()
    # plt.suptitle(f"{s}---Chunk ")
    # plt.show()
    # exit()
    
    # chunk["spectrogram"] = as_xarray_spectrogram(*compute_multispectrogram(chunk["filtered_song"].to_numpy(), chunk["t_song_coords"].data))
    # chunk["t_bin"] = chunk["t_bin_coords"].data.arr
    # print(chunk["spectrogram"].where(chunk["spectrogram"].notnull(), drop=True))
    # np.log(chunk["spectrogram"]).plot.pcolormesh()
    # plt.title(f"s={s}, chunk={c}")
    # plt.show()
    # exit()
    # d = regsel(d, t_song=slice(0, 30))
    # d["spectrogram"] = as_xarray_spectrogram(*compute_multispectrogram(d["filtered_song"].to_numpy(), d["t_song_coords"].data))
    # d = d[["spectrogram"]]
    # d = d.sel(f=slice(get_param("min_song_f")-get_param("f_spectrogram_shoulders"), get_param("max_song_f")+get_param("f_spectrogram_shoulders")))


    # f, axs = plt.subplots(3, sharex=True, sharey=True)



    # other_t, other_f, other_spec = usual_spectrogram(d["filtered_song"].to_numpy(), d["t_song_coords"].data)
    # other_arr = xr.DataArray(other_spec, dims=["f", "t"])
    # other_arr["t"] = other_t
    # other_arr["f"] = other_f
    # other_arr=other_arr.sel(f=slice(min_song_freq-100, max_song_freq+100))
    # other_arr.plot.pcolormesh(ax=axs[1])
    
    # axs[2].specgram(d["filtered_song"].to_numpy()[0:10**6], Fs=d["t_song_coords"].data.fs)
    # # axs[2].colorbar()

    # out_times, freqs, spectrogram = compute_multispectrogram(d["filtered_song"].to_numpy(), d["t_song_coords"].data)
    # d["t_analysis"]= xr.DataArray(out_times.arr, dims="t_analysis")
    # d["t_analysis_coords"] = xr.DataArray(out_times, dims="t_analysis")
    # d["freqs"] = xr.DataArray(freqs, dims="freqs")
    # d["spectrogram"] = xr.DataArray(spectrogram, dims=["freqs", "t_analysis"])
    # d = d.sortby("freqs")
    # d=d.sel(freqs=slice(min_song_freq-100, max_song_freq+100))
    
    # d["spectrogram"].plot.pcolormesh(ax=axs[0])
    
    
    # plt.show()
    # print(d)
    chunk = chunk.set_coords("t_bin_coords")
    chunk = chunk.drop_vars("t_bin")
    return chunk[["song_spectrogram"]]


# ============================================================================ #
# Run
# ============================================================================ #

if __name__ == "__main__":
    stop_on_error=True
    pd.Series.buffered_errors=not stop_on_error
    write_output=True
    parralelize = False

    def requires(session):
        return [get_session_storage_path(session)/"song/song_filtered.pkl"]

    def generates(session) -> Path:
        return get_session_storage_path(session)/f"song/song_filtered_spectrogram.pkl"

    # def get_chunk(session):
    #     coords = load_pickle(get_session_storage_path(session)/"metadata.pkl")["t_song_coords"].data
    #     n_chunks = int(np.ceil((coords.end - coords.start)/get_param("spectrogram_chunk_size_seconds")))
    #     return list(np.arange(n_chunks))

    sessions = retrieve_tasks(requires, generates, recompute_existing=True, n_per_group=1)

    errors = []
    last_bird = None
    def task(session):
        # global last_bird
        s, = session
        # if last_bird == s:
        #     return
        # else:
        #     last_bird = s
        # print(s)
        # if s=="B60/Sessions/2022-04-19/2022-04-19_11-03-08":
        #     return 
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
            e.add_note(f'During computation for task {s}')
            errors.append(e)
            if stop_on_error:
                raise
            else:
                logger.error(f'Error during computation for task {s}')
                raise e
        
    import concurrent.futures

    if parralelize:
        with concurrent.futures.ProcessPoolExecutor(5) as e:
            futures = list(tqdm.tqdm(e.map(task, sessions), total=len(sessions)))
    else:
        for s in tqdm.tqdm(sessions[:]):
            task(s)
    # for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(sessions)): pass
# for s,c in tqdm.tqdm(sessions):
#     try:
#         if len(requires(s)) > 0:
#             d = xr.merge([load_pickle(f) for f in requires(s)])
#         else:
#             d= xr.Dataset()

#         res = computation(d, s, c)
#         if write_output:
#             out = generates(s, c)
#             out.parent.mkdir(exist_ok=True, parents=True)
#             tmp_file = out.parent/ (out.stem + ".tmp"+out.suffix)
#             with  tmp_file.open("wb") as f:
#                 pickle.dump(res, f)
#             tmp_file.rename(out)
#         else:
#             print(res)
#     except Exception as e:
#         e.add_note(f'During computation for task {s, c}')
#         errors.append(e)
#         if stop_on_error:
#             raise
#         else:
#             logger.error(f'Error during computation for task {s, c}')

    if len(errors) > 0:
        if len(errors) > 1:
            raise ExceptionGroup(f"Found {len(errors)} errors", errors)
        else:
            raise errors[0]

