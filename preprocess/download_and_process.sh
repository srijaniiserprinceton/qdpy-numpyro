#/bin/sh
echo "[1.] Fetching data from JSOC ..."
python fetch_hmi_2drls.py
echo "[2.] Formatting frequency data to in and out files"
python format_freqdata.py
echo "[3.] Computing rotation and error profiles"
python rotation.py --smax 19
