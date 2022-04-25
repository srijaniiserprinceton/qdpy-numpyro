#/bin/sh
INSTR=hmi
echo "[0.] Generating the daylist ..."
python create_daylist.py --instrument $INSTR > temp.out 2> temp.err
echo "[1.] Fetching data from JSOC ..."
python fetch_data_2drls.py --instrument $INSTR
echo "[2.] Formatting frequency data to in and out files"
python format_freqdata.py --instrument $INSTR
echo "[3.] Computing rotation and error profiles"
python rotation.py --instrument $INSTR --smax 9
