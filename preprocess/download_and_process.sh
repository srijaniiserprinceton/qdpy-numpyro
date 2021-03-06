#/bin/sh
INSTR=$(python read_instr.py)
SMAX=$(python read_smax.py)
echo "instrument = $INSTR; smax = $SMAX"
echo "[0.] Generating the daylist ..."
python create_daylist.py --instrument $INSTR > temp.out 2> temp.err
echo "[1.] Fetching data from JSOC ..."
python fetch_data_jsoc.py --instrument $INSTR --wsr 1 --splits 1 --startidx 0 --endidx 73
echo "[2.] Cleaning fetched data ..."
python clean_fetched_data.py
echo "[3.] Formatting frequency data to in and out files"
python format_freqdata.py --instrument $INSTR
echo "[4.] Computing rotation and error profiles"
python rotation.py --instrument $INSTR --smax $SMAX
