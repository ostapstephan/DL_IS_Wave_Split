#!/bin/bash

ssh dlis wget -P /home/ubuntu/audiobooks $1;
scp dlis:/home/ubuntu/audiobooks/* /media/ostap/166E2FAB6E2F831B/audiobooks;
sleep 1;
ssh dlis rm /home/ubuntu/audiobooks/* ;



