#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="$HOME/HighzDrive"
LOGDIR="$HOME/.local/share/rclone"
LOGFILE="$LOGDIR/rclone-mount.log"
LOCKFILE="$HOME/.cache/rclone/mount_highz.lock"

mkdir -p "$MOUNTPOINT" "$LOGDIR" "$HOME/.cache/rclone"

# If your rclone.conf is encrypted and prompts for a password,
# uncomment and set this (or export it in ~/.bashrc)
# export RCLONE_CONFIG_PASS="your_config_password"

# If already mounted, do nothing
if findmnt -n -o FSTYPE "$MOUNTPOINT" | grep -q 'fuse.rclone'; then
  echo "Already mounted at $MOUNTPOINT"
  exit 0
fi

# Prevent concurrent mounts
exec 9>"$LOCKFILE"
flock -n 9 || { echo "Another mount attempt is in progress."; exit 1; }

echo "Mounting googledrive:high-z → $MOUNTPOINT (daemon)"
rclone mount "googledrive:high-z" "$MOUNTPOINT" \
  --vfs-cache-mode full \
  --vfs-write-back 10s \
  --vfs-cache-max-age 24h \
  --vfs-cache-max-size 3G \
  --cache-dir "$HOME/.cache/rclone" \
  --dir-cache-time 96h \
  --poll-interval 15s \
  --buffer-size 32M \
  --uid "$(id -u)" --gid "$(id -g)" --umask 022 \
  --drive-stop-on-upload-limit \
  --log-file "$LOGFILE" \
  --log-level INFO \
  --daemon

echo "Mount started. Logs → $LOGFILE"