from sunpy.net import jsoc
from sunpy.net import attrs as a
import jsoc_params as jsp
import argparse
import pandas as pd
import numpy as np
import os
import sys
import drms
import time

#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
INSTR = dirnames[-1]
ipdir = f"{scratch_dir}/input_files"
instrdir = f"{ipdir}/{INSTR}"
dldir = f"{instrdir}/dlfiles"
splitdir = f"{dldir}/splitdir"

if not os.path.isdir(instrdir): os.mkdir(instrdir)
if not os.path.isdir(dldir): os.mkdir(dldir)
if not os.path.isdir(splitdir): os.mkdir(splitdir)
#----------------------------------------------------------------------#
try:
    with open(f"{current_dir}/.jsoc_config", "r") as f:
        jsoc_config = f.read().splitlines()
    user_email = jsoc_config[0]
except FileNotFoundError:
    print(f"Please enter JSOC registered email in {current_dir}/.jsoc_config")
    sys.exit()
#-----------------------------------------------------------------------#
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
PARSER.add_argument("--tslen", help="72d or 360d",
                    type=str, default="72d")
PARSER.add_argument("--startidx", help="idx for daynum start",
                    type=int, default=0)
PARSER.add_argument("--endidx", help="idx for daynum end",
                    type=int, default=40)
PARSER.add_argument("--wsr", help="download wsr",
                    type=bool, default=0)
PARSER.add_argument("--splits", help="download splits",
                    type=bool, default=0)
ARGS = PARSER.parse_args()
del PARSER
#-----------------------------------------------------------------------#
INSTR = ARGS.instrument
ds_idx = ARGS.startidx
de_idx = ARGS.endidx

params = jsp.jsocParams(instr=INSTR)
series_name = params.series_name
daylist_fname = params.daylist_fname
LMAX = params.LMAX
NDT = params.NDT
freq_series_name = params.freq_series_name

drms_client = drms.Client()
segment_info = drms_client.info(series_name)
segments = segment_info.segments.index.values
#----------------------------------------------------------------------#
pt = pd.read_table(f'{daylist_fname}', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})
print(pt.shape)

day1 = pt['DATE'][ds_idx]
day2 = pt['DATE'][de_idx]
print(f"day1 = {day1}; day2 = {day2}")
print(f"atime = {a.Time(f'{day1}T00:00:00', f'{day2}T00:00:00')}")
"""
if ARGS.wsr:
    #----------------------------------------------------------------------#
    client = jsoc.JSOCClient()
    response = client.search(a.Time(f'{day1}T00:00:00', f'{day2}T00:00:00'),
                             a.jsoc.Series(series_name),
                             a.jsoc.Segment(segments[0]),
                             a.jsoc.Segment(segments[1]),
                             a.jsoc.Segment(segments[2]),
                             a.jsoc.Segment(segments[3]),
                             a.jsoc.PrimeKey('LMIN', '0') &
                             a.jsoc.PrimeKey('LMAX', LMAX) &
                             a.jsoc.PrimeKey('NDT', NDT) &
                             a.jsoc.PrimeKey('NACOEFF', '6') &
                             a.jsoc.PrimeKey('RADEXP', '-6') &
                             a.jsoc.PrimeKey('LATEXP', '-2'),
                             a.jsoc.Notify(user_email))
    print(f"Requesting data...")
    requests = client.request_data(response)
    #----------------------------------------------------------------------#
    count = 0
    while requests.status > 0:
        time.sleep(3)
        requests = client.request_data(response)
        print(f"request ID = {requests.id}; status = {requests.status}")
        if count > 5:
            print(f"Wait count = {count}. Trying to download")
            break
        count += 1
    print(f" status = {requests.status}: Ready for download")
    res = client.get_request(requests, path=dldir)
    #----------------------------------------------------------------------#

"""

if ARGS.splits:
    # ---- downloading the a-coefficient fits ---------------
    client = jsoc.JSOCClient()
    response = client.search(a.Time(f'{day1}T00:00:00', f'{day2}T00:00:00'),
                             a.jsoc.Series(f'{freq_series_name}'),
                             a.jsoc.PrimeKey('LMIN', '0') &
                             a.jsoc.PrimeKey('LMAX', LMAX) &
                             a.jsoc.PrimeKey('NDT', NDT) &
                             a.jsoc.Notify(user_email))
    print(f"Requesting data...")
    requests = client.request_data(response)
    #----------------------------------------------------------------------#
    count = 0
    while requests.status > 0:
        time.sleep(3)
        requests = client.request_data(response)
        print(f"request ID = {requests.id}; status = {requests.status}")
        if count > 5:
            print(f"Wait count = {count}. Trying to download")
            break
        count += 1
    print(f" status = {requests.status}: Ready for download")
    res = client.get_request(requests, path=splitdir)
    #----------------------------------------------------------------------#
