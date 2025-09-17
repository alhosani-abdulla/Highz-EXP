#!/usr/bin/env bash

set -euo pipefail
MOUNTPOINT="$HOME/HighzDrive"

if findmnt -T "$MOUNTPOINT" >/dev/null 2>&1; then
  fusermount -u "$MOUNTPOINT" 2>/dev/null || umount -l "$MOUNTPOINT"
  echo "Unmounted $MOUNTPOINT"
else
  echo "Not mounted: $MOUNTPOINT"
fi