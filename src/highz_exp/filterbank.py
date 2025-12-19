"""
Filterbank data loading and processing utilities for Highz-EXP

Provides convenient functions for loading FITS files, applying calibration,
and extracting spectrometer data.
"""

import os
import numpy as np
from astropy.io import fits
from pathlib import Path


class FilterbankData:
    """Container for filterbank spectral data"""
    
    def __init__(self, filename):
        """
        Load a filterbank FITS file
        
        Parameters
        ----------
        filename : str
            Path to FITS file
        """
        self.filename = filename
        self.data = None
        self.header = None
        self.frequencies = None
        self.voltages = None
        self.powers = None
        self.filter_channels = None
        self.metadata = {}
        
        self._load_fits()
    
    def _load_fits(self):
        """Load FITS file and extract data"""
        try:
            with fits.open(self.filename) as hdul:
                self.data = hdul[1].data
                self.header = hdul[1].header
                
                # Extract metadata
                self.metadata['voltage'] = self.header.get('SYSVOLT', 0.0)
                self.metadata['state'] = self.data[0][4] if len(self.data) > 0 else None
                self.metadata['num_sweeps'] = len(self.data)
                self.metadata['timestamp'] = os.path.getmtime(self.filename)
                
        except Exception as e:
            print(f"Error loading FITS file {self.filename}: {e}")
            raise
    
    def get_sweep(self, sweep_index, calibration=None):
        """
        Extract a single frequency sweep
        
        Parameters
        ----------
        sweep_index : int
            Index of sweep to extract (0 to num_sweeps-1)
        calibration : dict, optional
            Per-filter calibration dict with 'slope' and 'intercept'
        
        Returns
        -------
        tuple
            (frequencies, voltages, powers, filter_channels)
        """
        if self.data is None or sweep_index >= len(self.data):
            raise ValueError(f"Invalid sweep index {sweep_index}")
        
        # This requires access to calibration_utils from rtviewer
        # For now, return raw data with frequencies computed
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "highz-filterbank" / "tools" / "rtviewer"))
            import calibration_utils as cal
        except ImportError:
            # Fallback without calibration utility
            cal = None
        
        row = self.data[sweep_index]
        lo_freq = int(float(row[5]))  # FREQUENCY column
        
        if cal:
            a1 = row[0][:7]
            a2 = row[1][:7]
            a3 = row[2][:7]
            combined_ints = cal.makeSingleListOfInts(a1, a2, a3)
            voltages = np.array(cal.toVolts(combined_ints))
        else:
            # Return zeros if calibration unavailable
            voltages = np.zeros(21)
        
        # Compute frequencies for 21 filters
        frequencies = np.array([2.6 * x + 904 - lo_freq for x in range(21)])
        filter_channels = np.arange(21)
        
        # Apply calibration if provided
        if calibration:
            powers = []
            for filt_num, voltage in enumerate(voltages):
                if filt_num in calibration:
                    slope = calibration[filt_num]['slope']
                    intercept = calibration[filt_num]['intercept']
                    power = slope * voltage + intercept
                else:
                    power = -43.5 * voltage + 24.98  # Fallback
                powers.append(power)
            powers = np.array(powers)
        else:
            # Simple fallback calibration
            powers = -43.5 * voltages + 24.98
        
        return frequencies, voltages, powers, filter_channels
    
    def get_all_data(self, calibration=None):
        """
        Extract all sweeps into single arrays
        
        Parameters
        ----------
        calibration : dict, optional
            Per-filter calibration dict
        
        Returns
        -------
        dict
            Dictionary with keys 'frequencies', 'voltages', 'powers', 
            'filter_channels', 'metadata'
        """
        all_frequencies = []
        all_voltages = []
        all_powers = []
        all_filters = []
        
        for sweep_idx in range(self.metadata['num_sweeps']):
            freq, volt, power, filt = self.get_sweep(sweep_idx, calibration)
            all_frequencies.extend(freq)
            all_voltages.extend(volt)
            all_powers.extend(power)
            all_filters.extend(filt)
        
        return {
            'frequencies': np.array(all_frequencies),
            'voltages': np.array(all_voltages),
            'powers': np.array(all_powers),
            'filter_channels': np.array(all_filters),
            'metadata': self.metadata
        }
    
    def get_power_by_filter(self, calibration=None):
        """
        Get power readings organized by filter channel
        
        Returns
        -------
        dict
            {filter_num: array_of_power_readings}
        """
        data = self.get_all_data(calibration)
        powers_by_filter = {}
        
        for filt_idx in range(21):
            mask = data['filter_channels'] == filt_idx
            powers_by_filter[filt_idx] = data['powers'][mask]
        
        return powers_by_filter
    
    def summary(self):
        """Print summary of data"""
        print(f"\nFilterbank Data Summary")
        print(f"{'='*50}")
        print(f"File: {os.path.basename(self.filename)}")
        print(f"Sweeps: {self.metadata['num_sweeps']}")
        print(f"System Voltage: {self.metadata['voltage']:.2f} V")
        print(f"State: {self.metadata['state']}")
        print(f"Total data points: {self.metadata['num_sweeps'] * 21}")
        print(f"{'='*50}\n")


def load_filterbank_file(filepath, calibration=None):
    """
    Convenience function to load a filterbank FITS file
    
    Parameters
    ----------
    filepath : str
        Path to FITS file
    calibration : dict, optional
        Per-filter calibration dictionary
    
    Returns
    -------
    FilterbankData
        Object with data access methods
    """
    return FilterbankData(filepath)


def find_files_by_date(base_dir, date_str):
    """
    Find all FITS files for a specific date
    
    Parameters
    ----------
    base_dir : str
        Base directory containing date folders (e.g., Bandpass/)
    date_str : str
        Date string in format MMddyyyy (e.g., '11062025')
    
    Returns
    -------
    list
        List of FITS file paths
    """
    date_path = os.path.join(base_dir, date_str)
    
    if not os.path.exists(date_path):
        print(f"Date folder not found: {date_path}")
        return []
    
    files = sorted([
        os.path.join(date_path, f) for f in os.listdir(date_path)
        if f.endswith('.fits') and os.path.getsize(os.path.join(date_path, f)) > 0
    ])
    
    return files


def list_available_dates(base_dir):
    """
    List all available dates in data directory
    
    Parameters
    ----------
    base_dir : str
        Base directory containing date folders
    
    Returns
    -------
    list
        Sorted list of date strings (MMddyyyy format)
    """
    dates = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return []
    
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path) and len(entry) == 8 and entry.isdigit():
            dates.append(entry)
    
    return sorted(dates, reverse=True)


# Example usage
if __name__ == '__main__':
    # Example: Load a file and inspect data
    base_dir = "/Users/abdullaalhosani/Projects/highz/Data/LunarDryLake/2025Nov/filterbank/Bandpass"
    
    # List available dates
    dates = list_available_dates(base_dir)
    print(f"Available dates: {dates[:5]}")
    
    if dates:
        # Get files for first date
        files = find_files_by_date(base_dir, dates[0])
        print(f"Files on {dates[0]}: {len(files)}")
        
        if files:
            # Load first file
            fb = load_filterbank_file(files[0])
            fb.summary()
            
            # Get all data
            data = fb.get_all_data()
            print(f"Data shape: {len(data['frequencies'])} points")
            print(f"Power range: {data['powers'].min():.1f} to {data['powers'].max():.1f} dBm")
