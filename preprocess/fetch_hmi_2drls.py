from sunpy.net import jsoc
from sunpy.net import attrs as a
import pandas as pd
import numpy as np
import os
import drms
import time

#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
dldir = f"{ipdir}/hmi"
#----------------------------------------------------------------------#
series_name = "hmi.V_sht_2drls"
user_email = "g.samarth@tifr.res.in"

drms_client = drms.Client()
segment_info = drms_client.info(series_name)
segments = segment_info.segments.index.values

pt = pd.read_table(f'{ipdir}/daylist.txt', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

#----------------------------------------------------------------------#
day1 = pt['DATE'][0]  #[i+30]
day2 = pt['DATE'][1] #"2021-31-12"   #pt['DATE'][i+30]
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
                         a.jsoc.PrimeKey('LMAX', '200') &
                         a.jsoc.PrimeKey('NDT', '138240') &
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
