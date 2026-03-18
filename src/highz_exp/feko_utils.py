#!/usr/bin/env python3

# This module contains utilities for loading and processing antenna gain patterns from FEKO simulations, 
# as well as visualizing them.
import numpy as np
import pandas as pd
import healpy as hp
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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
    
    def load_gain_pattern(self, freq_hz):
        """
        Load the gain pattern for a specific frequency from the gain_info DataFrame.
        
        Parameters:
        freq_hz (float): The frequency in Hz for which to load the gain pattern.
        
        Returns:
        theta (np.ndarray): Incident angles in degrees.
        gain (np.ndarray): 2D Gain map with (theta, phi) dimensions.
        """
        # Filter the DataFrame for the specified frequency
        freq_data = self.gain_info[self.gain_info['Frequency (Hz)'] == freq_hz]
        
        # Extract theta and gain values
        theta = np.deg2rad(freq_data['IncidentTheta (deg)'].to_numpy())
        gain = freq_data['Voltage_Mag'].to_numpy().reshape((len(theta), 1)).repeat(181, axis=1)  # Using Voltage_Mag as a proxy for gain
        
        return theta, gain

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
    @staticmethod
    def generate_beam_map(gain_map) -> np.ndarray:
        """
        Generate a beam map of the antenna with a given gain map using interpolation.
        
        Parameters:
        - gain_map: 2D array of gain values with shape (num_theta, num_phi)
        Returns:
        - beam_map: 1D array of gain values for each HEALPix pixel, with the same ordering as hp.pix2ang
        """
        num_theta, num_phi = gain_map.shape
        theta1 = np.linspace(0, np.pi/2, num_theta)
        phi1 = np.linspace(0, 2*np.pi, num_phi)
        interpolator = RegularGridInterpolator(
            (theta1, phi1), gain_map,
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