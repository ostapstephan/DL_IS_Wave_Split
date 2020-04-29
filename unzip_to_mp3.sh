#!/bin/bash

audio_path= '/media/ostap/166E2FAB6E2F831B/audiobooks'
mp3_path= '/media/ostap/166E2FAB6E2F831B/audiobooks/mp3/'

find /media/ostap/166E2FAB6E2F831B/audiobooks -type f > unzip_list.txt
cat unzip_list.txt | xargs -n 1 -P 7 -I{} unzip -o {} -d /media/ostap/166E2FAB6E2F831B/audiobooks/mp3/


