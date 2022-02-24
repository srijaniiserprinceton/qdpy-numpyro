from qdpy import jax_functions as jf
import os

os.system(f"ls summary* > fnames.txt")
with open(f"fnames.txt", "r") as f:
    fnames = f.read().splitlines()

for fname in fnames:
    _fname = fname[:-4]
    print(f"Updating {_fname}")
    summary_data = jf.load_obj(_fname)
    _gvars_dpy = summary_data['params']['dpy']['GVARS']
    try:
        _gvars_qdpy = summary_data['params']['qdpy']['GVARS']
        summary_data['params']['qdpy']['GVARS'] = jf.dict2obj(_gvars_qdpy.__dict__)
        _gvars_qdpy = summary_data['params']['qdpy']['GVARS']
    except KeyError:
        _gvars_qdpy = _gvars_dpy
        pass
    summary_data['params']['dpy']['GVARS'] = jf.dict2obj(_gvars_dpy.__dict__)
    _gvars_dpy = summary_data['params']['dpy']['GVARS']
    print(f" -- [after ] {_gvars_dpy}, {_gvars_qdpy}")
    jf.save_obj(summary_data, _fname)
