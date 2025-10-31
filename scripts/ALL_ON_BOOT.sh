#!/bin/sh
# /home/peterson/ALL_ON_BOOT.sh

LOGDIR="/home/peterson/logs"
mkdir -p "$LOGDIR"

echo "[$(date)] Starting Digital Spectrometer Setup..." >> "$LOGDIR/digital_spec.log"
/home/peterson/Highz-EXP/src/digital_spectrometer/launcher.sh >> "$LOGDIR/digital_spec_cronlog_$(date +\%Y-\%m-\%d_\%H-\%M-\%S).log" 2>&1

#echo "[$(date)] Starting Daemon Setup..." >> "$LOGDIR/daemon_setup.log"
#/home/peterson/launcher_Daemon_Setup.sh >> "$LOGDIR/daemon_setup.log" 2>&1

#echo "[$(date)] Starting FB process..." >> "$LOGDIR/fb.log"
#/home/peterson/launcher_FB.sh >> "$LOGDIR/fb.log" 2>&1

echo "[$(date)] All startup processes complete." >> "$LOGDIR/master.log"
