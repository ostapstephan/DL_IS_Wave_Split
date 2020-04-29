#!/bin/bash
# latch onto a running process and wait till it's done then run the stuff below 

while [ -e /proc/$1 ]; do
    echo -n "."  # Do something while the background command runs.
    sleep 300  # Optional: slow the loop so we don't use up all the dots.
done


cp meta.csv /share/audiobooks
echo 'done'

python add_speaker_id.py
sleep 10
echo 'started converting'
python convert_wav_2_tfrec.py

