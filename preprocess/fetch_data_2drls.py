from sunpy.net import jsoc
from sunpy.net import attrs as a
import argparse
import pandas as pd
import numpy as np
import os
import sys
import drms
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
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
ARGS = PARSER.parse_args()
del PARSER
#-----------------------------------------------------------------------#
INSTR = ARGS.instrument
#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
instrdir = f"{ipdir}/{INSTR}"
dldir = f"{instrdir}/dlfiles"

if not os.path.isdir(instrdir): os.mkdir(instrdir)
if not os.path.isdir(dldir): os.mkdir(dldir)
#----------------------------------------------------------------------#
if INSTR=="hmi":
    series_name = "hmi.V_sht_2drls"
    daylist_fname = "daylist.hmi"
    LMAX = "200"
    NDT = "138240"
elif INSTR=="mdi":
    series_name = "mdi.vw_V_sht_2drls"
    daylist_fname = "daylist.mdi"
    LMAX = "300"
    NDT = "103680"

drms_client = drms.Client()
segment_info = drms_client.info(series_name)
segments = segment_info.segments.index.values

pt = pd.read_table(f'{daylist_fname}', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

#----------------------------------------------------------------------#
# day1 = pt['DATE'][0]  #[i+30]
# day2 = pt['DATE'][1] #"2021-31-12"   #pt['DATE'][i+30]
day1 = pt['DATE'][0]  #[i+30]
day2 = pt['DATE'][1]  #[i+30]
print(f"day1 = {day1}; day2 = {day2}")
print(f"atime = {a.Time(f'{day1}T00:00:00', f'{day2}T00:00:00')}")
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
os.system(f"cd {dldir}; rm $(ls | egrep -v '{NDT}.6')")
# os.system(f"cd {dldir}; rm $(ls | egrep -v '{NDT}.36')")
#----------------------------------------------------------------------#
