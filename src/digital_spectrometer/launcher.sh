#!/bin/sh
#launcher.sh
#navigate to home directory, then to this directory, then execute python script, then back home

cd /
cd home/peterson/Highz-EXP/src/digital_spectrometer
/home/peterson/cfpga_venv/bin/python run_spectrometer.py
cd /
