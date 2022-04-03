from datetime import datetime as DateTime, timedelta as TimeDelta
import os
import numpy as np
import pandas as pd
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
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

def convert2pdformat(newdate):
    yr = newdate.year
    month = newdate.month
    day = newdate.day
    datestr = f"{yr}-{month:02d}-{day:02d}"
    return datestr

MDIDAYS = []
DATES = []
SNS = []
print("   ")

if INSTR=="hmi":
    newdate = DateTime(2010, 4, 30)
    yearmax = 2022
    newmdi = 6328
    tslen = 72

elif INSTR=="mdi":
    # newdate = DateTime(1996, 5, 1)
    # newdate = DateTime(1993, 1, 1)
    mdiepoch = DateTime(2010, 4, 30) - TimeDelta(days=6328)
    newdate = DateTime(1996, 5, 1)
    yearmax = 2012
    newmdi = (newdate - mdiepoch).days
    tslen = 72

newsn = 0
year = newdate.year
datestr = convert2pdformat(newdate)

MDIDAYS.append(newmdi)
DATES.append(datestr)
SNS.append(newsn)

firstmdi = newmdi
firstsn = newsn
firstdate = newdate

# while year > yearmin:
while year < yearmax:
    newdate = firstdate + TimeDelta(days=tslen)
    newmdi = firstmdi + tslen
    newsn = firstsn + 1
    year = newdate.year
    datestr = convert2pdformat(newdate)

    MDIDAYS.append(newmdi)
    DATES.append(datestr)
    SNS.append(newsn)

    firstmdi = newmdi
    firstsn = newsn
    firstdate = newdate
    print(f"{newmdi:5d} :: {datestr}")
    if datestr=='1998-04-21' and INSTR=="mdi":
        print("HERE")
        resetdate = DateTime(1999, 2, 3)
        newmdi = firstmdi + (resetdate - newdate).days
        newsn = firstsn + 1
        datestr = convert2pdformat(resetdate)
        MDIDAYS.append(newmdi)
        DATES.append(datestr)
        SNS.append(newsn)

        firstmdi = newmdi
        firstsn = newsn
        firstdate = resetdate

d = {'SN': SNS,
     'MDI': MDIDAYS,
     'DATE': DATES}
pt2 = pd.DataFrame(data=d)
pt2.to_csv(f'daylist.{INSTR}', sep='\t', index=False, header=False)
