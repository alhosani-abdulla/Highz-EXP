# Scripts Directory

This directory contains shell scripts for managing the HighZ-EXP experimental setup, including system initialization, file mounting, and data management.

## Scripts Overview

### `ALL_ON_BOOT.sh`
**Purpose:** System startup script for automated initialization of the digital spectrometer

**Description:**
- Automatically runs on system boot to initialize the digital spectrometer setup
- Starts the digital spectrometer launcher with proper logging
- Logs all operations to timestamped files in `/home/peterson/logs/`
- Contains commented sections for additional daemon and FB processes

**Usage:**
```bash
sudo ./ALL_ON_BOOT.sh
```

**Log Files:**
- Main log: `/home/peterson/logs/digital_spec.log`
- Timestamped logs: `/home/peterson/logs/digital_spec_No_HPLP_two_shorts_cronlog_YYYY-MM-DD_HH-MM-SS.log`
- Master log: `/home/peterson/logs/master.log`

### `mount_highz.sh`
**Purpose:** Mount Google Drive storage for HighZ experiment data

**Description:**
- Mounts the `googledrive:high-z` remote to `$HOME/HighzDrive`
- Uses rclone with optimized VFS caching for large data files
- Includes safety checks to prevent duplicate mounts
- Runs as a background daemon with comprehensive logging

**Features:**
- **VFS Cache Mode:** Full caching for optimal performance
- **Write-back:** 10-second delay for write operations
- **Cache Settings:** 24-hour max age, 3GB max size
- **Concurrent Mount Protection:** Uses file locking to prevent conflicts

**Usage:**
```bash
./mount_highz.sh
```

**Prerequisites:**
- rclone must be installed and configured
- Google Drive remote named `googledrive` must be set up in rclone config
- For encrypted configs, set `RCLONE_CONFIG_PASS` environment variable

### `unmount_highz.sh`
**Purpose:** Safely unmount the HighZ Google Drive

**Description:**
- Cleanly unmounts the Google Drive from `$HOME/HighzDrive`
- Uses `fusermount` for graceful unmounting with fallback to lazy unmount
- Includes status checking to verify mount state before attempting unmount

**Usage:**
```bash
./unmount_highz.sh
```

## Setup Instructions

### 1. Make Scripts Executable
```bash
chmod +x *.sh
```

### 2. Configure rclone (for mount scripts)
```bash
rclone config
# Follow prompts to set up Google Drive remote named "googledrive"
```

### 3. Set up Boot Script (optional)
To run `ALL_ON_BOOT.sh` automatically on system startup:

**Using systemd:**
```bash
sudo cp ALL_ON_BOOT.sh /usr/local/bin/
sudo nano /etc/systemd/system/highz-boot.service
```

Add the following content:
```ini
[Unit]
Description=HighZ Experiment Boot Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ALL_ON_BOOT.sh
User=peterson
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl enable highz-boot.service
```

## File Structure

```
scripts/
├── README.md              # This file
├── ALL_ON_BOOT.sh         # System startup script
├── mount_highz.sh         # Google Drive mount script
└── unmount_highz.sh       # Google Drive unmount script
```

## Troubleshooting

### Mount Issues
- **Problem:** `rclone mount` fails with authentication error
  - **Solution:** Run `rclone config reconnect googledrive` to refresh tokens

- **Problem:** Mount point already in use
  - **Solution:** Run `./unmount_highz.sh` first, then retry mounting

### Permission Issues
- **Problem:** Scripts fail with permission denied
  - **Solution:** Ensure scripts are executable with `chmod +x *.sh`

### Log Analysis
Check log files for detailed error information:
```bash
# Recent mount logs
tail -f ~/.local/share/rclone/rclone-mount.log

# Boot script logs
tail -f /home/peterson/logs/digital_spec.log
```

## Dependencies

- **rclone:** Required for Google Drive mounting
- **fuse:** Required for filesystem mounting
- **bash:** Required for script execution
- **systemd:** Optional, for automatic boot execution

## Notes

- The mount script uses aggressive caching settings optimized for large scientific data files
- Log rotation should be implemented for production use to prevent disk space issues
- Scripts are designed for the specific user `peterson` and paths - modify as needed for your setup