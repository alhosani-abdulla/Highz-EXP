#!/usr/bin/env python3

# This module contains utilities for loading and processing antenna gain patterns from FEKO simulations, 
# as well as visualizing them.
import numpy as np
import pandas as pd
import healpy as hp
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

try:
    from tqdm.auto import tqdm
except ImportError:
    class _NullTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable or [])

        def update(self, *args, **kwargs):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _NullTqdm(iterable)

from highz_exp.plotter import generate_static_hp_map, visualize_static_hmap

# healpy variables
N_SIDE = 256
N_PIX = hp.nside2npix(N_SIDE)

class AntennaGain():
    """Class to handle loading and processing of antenna gain pattern output files from FEKO simulations.
    
    Note: This requires a specific format of the input"""
    def __init__(self, gain_info):
        """Initialize the AntennaGain object with gain information.
        
        Parameters:
            - gain_info (pd.DataFrame)
              DataFrame containing the gain information with columns 
              'Frequency (Hz)', 'IncidentTheta (deg)', and 'Voltage_Mag'.
        """
        self.gain_info = gain_info
        if not 'Voltage_Mag' in gain_info.columns:
            self.gain_info['Voltage_Mag'] = np.sqrt(gain_info['Voltage_Real (V)']**2 + gain_info['Voltage_Imag (V)']**2)
    
    def load_gain_pattern(self, freq_hz) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the gain pattern for a specific frequency from the gain_info DataFrame.
        
        Parameters:
        freq_hz (float): The frequency in Hz for which to load the gain pattern.
        
        Returns:
            theta (np.ndarray): Incident angles in degrees.
            gain (np.ndarray): 2D Gain map (effective height) with (theta, phi) dimensions.
        """
        # Filter the DataFrame for the specified frequency
        freq_data = self.gain_info[self.gain_info['Frequency (Hz)'] == freq_hz]
        
        # Extract theta and gain values
        theta = np.deg2rad(freq_data['IncidentTheta (deg)'].to_numpy())
        gain = freq_data['Voltage_Mag'].to_numpy().reshape((len(theta), 1)).repeat(181, axis=1)  # Using Voltage_Mag as a proxy for gain
        
        return theta, gain
    
    def load_max_gain(self) -> pd.DataFrame:
        """Load the maximum gain values for each frequency from the gain_info DataFrame.
        
        Returns:
            max_gain_df (pd.DataFrame): DataFrame containing all columns for rows with maximum gain per frequency.
        """
        max_gain_df = self.gain_info.loc[self.gain_info.groupby('Frequency (Hz)')['Voltage_Mag'].idxmax()]
        return max_gain_df
    
    def load_frequency_range(self) -> np.ndarray:
        """Load the unique frequency values from the gain_info DataFrame.
        
        Returns:
            frequencies (np.ndarray): Array of unique frequency values in Hz.
        """
        frequencies = self.gain_info['Frequency (Hz)'].unique()
        frequencies = np.sort(frequencies)  # Sort frequencies in ascending order
        return frequencies

    def eff_height_freq_plot(self, title, unit='Voltage') -> pd.DataFrame:
        """Create a plot of effective height vs frequency using the maximum gain values.
        
        Returns:
            max_gain_df (pd.DataFrame): DataFrame containing all columns for rows with maximum gain per frequency.
        """
        max_gain_df = self.load_max_gain()
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel('Frequency (MHz)')
        if unit == 'Voltage':
            plt.ylabel('Effective Height (V)')
            plt.plot(max_gain_df['Frequency (Hz)'] / 1e6, max_gain_df['Voltage_Mag'], marker='o')
        elif unit == 'Power':
            plt.ylabel('Effective Height (Power)')
            plt.plot(max_gain_df['Frequency (Hz)'] / 1e6, max_gain_df['Voltage_Mag']**2, marker='o')
        else:
            raise ValueError("Invalid unit. Use 'Voltage' or 'Power'.")
        plt.grid()
        plt.show()
        return max_gain_df

    def load_and_plot(self, freq_mhz) -> np.ndarray:
        """Load the gain pattern for a specific frequency, create a polar plot, and generate a beam map.
        Parameters:
            freq_mhz (float): The frequency in MHz for which to load and plot""" 
        # Load the gain pattern for the specified frequency and create a polar plot.
        theta, gain = self.load_gain_pattern(freq_mhz*1e6)
        self.create_polar_plot(theta, gain, title=f'Gain Pattern at {freq_mhz} MHz')

        # Generate the beam map and visualize it
        beam_map = self.generate_beam_map(gain)
        gain_phis = np.linspace(0, 2*np.pi, 181, endpoint=True)
        self.visualize_beam_map(beam_map)

        return beam_map
    
    # By Theo Dardio
    def create_simulated_waterfall(self, utc_timestamps, location) -> np.ndarray:
        """Create a simulated waterfall of antenna data based on healpix maps and beam patterns.
        Parameters:
            - utc_timestamps: List of UTC timestamps (datetime.datetime objects) for which to generate the data.
            - location: Tuple of (latitude, longitude) for the observer's location.
        Returns:
            - simulated_antenna_data: 2D numpy array of shape (len(utc_timestamps), len(frequencies_mhz)) containing the simulated antenna data.
        """
        frequencies_mhz = self.load_frequency_range() / 1e6
        # healpy variables
        n_side = 256
        n_pix = hp.nside2npix(n_side)
        # Solid angle of each pixel
        omega = hp.nside2pixarea(n_side)  # in steradians
        simulated_antenna_data = np.zeros((len(utc_timestamps), len(frequencies_mhz)))
        total_steps = len(utc_timestamps) * len(frequencies_mhz)
        progress = tqdm(total=total_steps, desc="Simulating waterfall", unit="step", colour="cyan")
        for j, freq in enumerate(tqdm(frequencies_mhz, desc="Frequencies", leave=False, colour="magenta")):
            effective_heights_2d_map = self.load_gain_pattern(freq_hz=freq*1e6)[1]
            beam_map = self.generate_beam_map(effective_heights_2d_map)
            D = np.sum(beam_map**2) * omega
            for i, timestamp in enumerate(utc_timestamps):
                hmap = generate_static_hp_map(frequency_mhz=freq, 
                        utc_timestamp=timestamp, location=location, observer='LFSM')
                N = np.sum(hmap * beam_map ** 2) * omega
                simulated_antenna_data[i, j] = N/D
            progress.update(1)
            progress.set_postfix(freq_mhz=f"{freq:.2f}", timestamp_index=i + 1, refresh=False)
        progress.close()
        return simulated_antenna_data

    # By Theo Dardio
    @staticmethod
    def generate_beam_map(effective_height_2d_map) -> np.ndarray:
        """
        Generate a beam map of the antenna with a given gain map using interpolation.
        
        Parameters:
        - effective_height_2d_map: 2D array of effective height values with shape (num_theta, num_phi)
        Returns:
        - beam_map: 1D array of length N_PIX containing the interpolated beam map values for each pixel
        """
        num_theta, num_phi = effective_height_2d_map.shape
        theta1 = np.linspace(0, np.pi/2, num_theta)
        phi1 = np.linspace(0, 2*np.pi, num_phi)
        interpolator = RegularGridInterpolator(
            (theta1, phi1), effective_height_2d_map,
            bounds_error=False, fill_value=0
        )
        beam_map = np.full(N_PIX, 0.0)  # initialize with UNSEEN for masking
        # Get pixel centers in (θ, φ)
        theta_hp, phi_hp = hp.pix2ang(N_SIDE, np.arange(N_PIX))
        # Find pixels in upper hemisphere (θ <= π/2)
        mask_upper = theta_hp <= (np.pi / 2)
        # Prepare interpolation points only for those pixels
        interp_points = np.vstack((theta_hp[mask_upper], phi_hp[mask_upper])).T
        # Interpolate
        interp_values = interpolator(interp_points)
        # Assign to map (fill rest with UNSEEN or 0)
        beam_map[mask_upper] = interp_values
        rot = hp.Rotator([0, 90, 0])
        beam_map = rot.rotate_map_pixel(beam_map)
        return beam_map
    
    # By Theo Dardio
    @staticmethod
    def visualize_beam_map(beam_map):
        hp.orthview(beam_map, half_sky=True, coord='C', title="Beam Map")
        # Now add custom labels for azimuth and zenith angle
        ax = plt.gca()

        # Add concentric zenith angle circles (like elevation rings)
        zenith_angles = [10, 30, 60, 80]
        for za in zenith_angles:
            circle = plt.Circle((0, 0), np.sin(np.radians(za)), color='white', ls='--', fill=False, alpha=0.5)
            ax.add_artist(circle)
            plt.text(0, np.sin(np.radians(za)) + 0.01, f"{za}°", color='white', ha='center')

        # Add azimuth angle labels
        
        az_labels = [0, 90, 180, 270]
        label_pos = {
            0: (0, 1.05),       # North (up)
            90: (1.05, 0),      # East (right)
            180: (0, -1.1),     # South (down)
            270: (-1.1, 0),     # West (left)
        }
        for az in az_labels:
            x, y = label_pos[az]
            plt.text(x, y, f"{az}°", color='white', ha='center', va='center')

        plt.show()
    
    # By Theo Dardio
    @staticmethod
    def create_polar_plot(gain_thetas, gain_values, title='Basic Polar Plot'):
        """Creates a polar plot of the effective height at one phi slice.
        
        Parameters:
        - gain_values: 2D array of gain values with shape (num_theta, num_phi)"""
        # Create polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Set 0° at top, clockwise
        ax.set_theta_zero_location('N')  # 0 degrees = North (top)
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.plot(gain_thetas, gain_values[:,0])  # Plot the first phi slice as an example

        plt.title(title)
        plt.show()