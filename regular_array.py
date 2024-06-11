
import numpy as np, xarray as xr



def add_numerics(klass):
    for numeric_fn in ['add','radd','sub','rsub','mul','rmul','truediv','floordiv']:
        dunder_fn = '__{}__'.format(numeric_fn)
        setattr(klass, dunder_fn, lambda self, *args, **kwargs: getattr(self.arr, dunder_fn)(*args, **kwargs))
    return klass

@add_numerics
class RegularArray:
    def __init__(self, start_index, fs, ndata, shift=0, dtype=float):
        self.start_index = start_index
        self.shift = shift
        self.fs=fs
        self.npoints=ndata
        self.dtype=np.dtype(dtype)

    @staticmethod
    def from_params(ndata=None, start=None, end=None, fs=None, d=None, border_right="<=", border_left="<<", shift=None, dtype=float):
        match d, fs:
            case None, None: pass
            case _, None: fs = 1/d
            case None, _: pass
            case _, _: raise Exception("Both fs and d cannot be specified")

        if fs is None:
            if start is not None and end is not None and ndata is not None:
                if border_right=="<=" and border_left =="<<":
                    fs = ndata/(end-start+1)
                    end=None
        
        
        if not fs is None:
            if shift is None:
                shift_start = start - np.floor(start/fs) * fs if not start is None else np.nan
                shift_end = end - np.floor(end/fs) * fs if not end is None else np.nan
                shift = np.nanmean([shift_start, shift_end])
                if np.isnan(shift):
                    shift=0
            def get_index(val, border):
                return RegularArray.get_index(val, shift, fs, border)

            if (not start is None) and (not end is None):
                if not ndata is None:
                    raise Exception("Two much specification for RegularArray.from_params")
                return RegularArray(get_index(start, border_right), fs, get_index(end, border_left)-get_index(start, border_right) +1, shift=shift, dtype=dtype)
            elif ndata is not None and start is not None:
                return RegularArray(get_index(start, border_right), fs, ndata, shift=shift, dtype=dtype)
            elif ndata is not None and end is not None:
                return RegularArray(get_index(end, border_left) +1 - ndata, fs, ndata, shift=shift, dtype=dtype)
            elif ndata is not None:
                return RegularArray(0, fs, ndata, shift=shift, dtype=dtype)  
        else:
            raise Exception("Either fs or d parameters must be specified")
    
    @staticmethod
    def get_index(val, shift, fs, border="~="):
        match border:
            case "<<": return int(np.ceil((val-shift)*fs)) -1
            case "<=": return int(np.floor((val-shift)*fs)) 
            case ">=": return int(np.ceil((val-shift)*fs))
            case ">>": return int(np.floor((val-shift)*fs)) +1
            case "~=": return int(np.round((val-shift)*fs))
    
    def get_relative_index(self, val, border="~="):
        return RegularArray.get_index(val, self.shift, self.fs, border) - self.start_index
    # @staticmethod
    # def from_interval(start, fs, ndata, dtype=float):
    #     r= RegularArray(start, fs, ndata, dtype=dtype)
    #     if r.size != ndata:
    #         raise Exception("error")
    #     return r
    
    @staticmethod
    def from_array(a, dtype=float):
        import sklearn.linear_model
        m = sklearn.linear_model.LinearRegression().fit(np.arange(a.size).reshape(-1, 1), a.reshape(-1))
        start = m.intercept_
        fs = 1/m.coef_[0]
        rounded_fs = np.round(fs*10**3)/10**3
        res = RegularArray.from_params(start=start, fs=rounded_fs, ndata=a.size, shift=None, dtype=dtype, border_left="~=", border_right="~=")
        if (np.abs(res.arr-a)*fs < 10**(-1)).all():
            return res
        else:
            raise Exception(f"not regular enough {res} {a} {(res.arr-a)*fs}")

    @property 
    def end(self):
        return (self.start_index + self.npoints)/self.fs + self.shift
    @property 
    def start(self):
        return (self.start_index)/self.fs + self.shift
    
    @property
    def arr(self):
        return np.arange(self.start_index, self.start_index+self.npoints)/self.fs + self.shift
    
    def __str__(self):
        return f"Regular(n={self.npoints}, fs={self.fs}, [{self.start}, {self.end}[)"
    def __repr__(self):
        return self.__str__()

    def __array_function__(self, *args, **kwargs):
        return self.arr.__array_function__(*args, **kwargs)
    def __array_ufunc__(self, *args, **kwargs):
        print(f"executing __array_ufunc__({args}, {kwargs})" )
        return self.arr.__array_ufunc__(*args, **kwargs)
    
    def __array__(self, *args, **kwargs):
        return self.arr.__array__(*args, **kwargs)

    def __getattr__(self, name):
        # print(f"called {name} {callable(getattr(self.arr, name))}")
        if callable(getattr(self.arr, name)):
            return getattr(self.arr, name)
        else:
            return getattr(self.arr, name)
    
    @property
    def shape(self):
        return (self.npoints, )
    @property
    def ndim(self):
        return 1
    @property
    def nbytes(self):
        return 0
    
    @property
    def name(self):
        return self.__str__()
    
    def _repr_inline_(self, l):
        # if len(self.__str__()) > l:
        #     min = len("linspace(, , fs=)")
        #     left_over = l-min
        #     s = (l-min)//3

        #     return f"linspace({self.start:.{s}f}, {self.end:.{s}f}, fs={self.fs:.{s}f}, end={self.endpoint:.{s}f})"
        # else:
        return self.__str__()
    
        
    def __getstate__(self) -> object:
        return dict(start_index=self.start_index, fs=self.fs, npoints=self.npoints, shift=self.shift, dtype=self.dtype)
    

    def __setstate__(self, state):
        self.start_index = state["start_index"]
        self.fs=state["fs"]
        self.dtype=state["dtype"]
        self.npoints = state["npoints"]
        self.shift = state["shift"]


def regsel(a: xr.DataArray, **dimkwargs):
    if len(dimkwargs)!=1:
        raise "Problem"
    dim = list(dimkwargs.keys())[0]
    val: slice = list(dimkwargs.values())[0]
    coord = a[f"{dim}_coords"].data
    a = a.drop_vars(f"{dim}_coords")
    n_start = coord.get_relative_index(val.start, "<=")
    n_end = coord.get_relative_index(val.stop, ">=")
    if n_start < 0:
        n_start=0
    if n_end > coord.npoints:
        n_end = coord.npoints
    # print(coord)
    # print(val, n_start, n_end)
    a = a.isel({dim: slice(n_start, n_end)})
    new_regar = RegularArray(start_index=coord.start + n_start, fs=coord.fs, ndata= n_end-n_start, shift=coord.shift, dtype=coord.dtype)
    # print(new_regar)
    a = a.assign_coords({f"{dim}_coords": xr.DataArray(new_regar, dims=dim)})
    return a
# RegularArray.from_start(2.1, 100, 10003)
# import pickle
# pickle.dump(RegularArray(0, 10, 100), open("stupid.pkl", "wb"))
# print(pickle.load(open("stupid.pkl", "rb")))