import numba
from pathlib import Path

@numba.jit(nopython=True)
def search(a, index, step=1, stop=None, fill=-1):
    if stop is None:
       stop = a.size if step > 0 else -1
    for i in range(index, int(stop), step):
        if a[i]:
            return i
    return fill

@numba.guvectorize([(numba.uint8[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:])], "(n), (), (), (), () -> ()", nopython=True)
def fastsearch(a, index, step, max_n_search, fill, res):
    n_search = 0
    curr_index= index
    res[:] = fill
    while curr_index >= 0 and curr_index < a.shape[-1] and (n_search < max_n_search or max_n_search < 0):
        if a[curr_index]:
            res[:]= curr_index
            break
        n_search+=1
        curr_index+=step



@numba.jit(nopython=True)
def subarray_positions(a, pattern):
    for i in range(a.size):
        r = True
        for j in range(pattern.size):
            if a[i+j] != pattern[j]:
                r=False
        if r:
            yield i
    
    

def singleglob(p: Path, *patterns, error_string='Found {n} candidates for pattern {patterns} in folder {p}', only_ok=False):
    all = [path for pat in patterns for path in p.glob(pat)]
    if only_ok:
        return len(all)==1
    if len(all) >1:
        raise Exception(error_string.format(p=p, n=len(all), patterns=patterns))
    if len(all) ==0:
        raise Exception(error_string.format(p=p, n=len(all), patterns=patterns))
    return all[0]
