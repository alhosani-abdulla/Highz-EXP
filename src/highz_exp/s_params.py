import numpy as np
from highz_exp.spec_proc import smooth_spectrum
import skrf as rf
import pickle
import os, copy
from matplotlib import pyplot as plt

from highz_exp.plotter import plot_gain

pjoin = os.path.join

class S_Params:
    def __init__(self, s_params_files=None, labels=None, ntwk_dict=None, pickle_file=None):
        """
        Construct S_Params in one of three ways:

        - Provide s_params_files (str or list) and optional labels (str or list)
        - Provide ntwk_dict: a dict of {label: rf.Network} or {label: filepath}
        - Provide pickle_file: path to a pickled dict of networks (same format as ntwk_dict)

        Parameters:
        - labels (str or list): Labels for the S-parameter files.
        """
        # Priority: ntwk_dict > pickle_file > s_params_files
        if ntwk_dict is not None:
            if not isinstance(ntwk_dict, dict):
                raise TypeError("ntwk_dict must be a dict mapping labels to rf.Network or file paths.")
            self.ntwk_dict = {}
            for label, val in ntwk_dict.items():
                if isinstance(val, rf.Network):
                    self.ntwk_dict[label] = val
                elif isinstance(val, str):
                    self.ntwk_dict[label] = rf.Network(val)
                else:
                    raise TypeError("ntwk_dict values must be skrf.Network instances or filepath strings.")
            return

        if pickle_file is not None:
            if not isinstance(pickle_file, str):
                raise TypeError("pickle_file must be a filepath string.")
            with open(pickle_file, "rb") as fh:
                loaded = pickle.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("Pickle must contain a dict of networks (label -> rf.Network or filepath).")
            # reuse ntwk_dict path
            self.__init__(ntwk_dict=loaded)
            return

        if s_params_files is None:
            raise ValueError("Must provide one of: ntwk_dict, pickle_file, or s_params_files (+ optional labels).")

        # Handle s_params_files and labels
        if isinstance(s_params_files, str):
            s_params_files = [s_params_files]
        if labels is None:
            labels = [os.path.splitext(os.path.basename(f))[0] for f in s_params_files]
        elif isinstance(labels, str):
            labels = [labels]

        if len(s_params_files) != len(labels):
            raise ValueError("Number of S-parameter files must match number of labels.")

        self.ntwk_dict = {label: rf.Network(file) for file, label in zip(s_params_files, labels)}

    @classmethod
    def from_files(cls, s_params_files, labels=None):
        return cls(s_params_files=s_params_files, labels=labels)

    @classmethod
    def from_ntwk_dict(cls, ntwk_dict):
        return cls(ntwk_dict=ntwk_dict)

    @classmethod
    def from_pickle(cls, pickle_file):
        return cls(pickle_file=pickle_file)
    
    def get_freq(self, MHz=True) -> np.ndarray:
        """
        Get frequency axis from the first Network in ntwk_dict.

        Parameters:
        - MHz (bool): If True, return frequency in MHz; else in Hz.

        Returns:
        - np.ndarray: Frequency axis.
        """
        if not self.ntwk_dict:
            raise ValueError("No networks available to extract frequency axis.")
        first_network = next(iter(self.ntwk_dict.values()))
        freq = first_network.f
        if MHz:
            freq = freq / 1e6
        return freq
    
    def get_s21(self, db=True) -> tuple[dict, dict]:
        """
        Load S-parameters from .s2p files in ntwk_dict, converting to dB if specified.

        Parameters:
        - db (bool): If True, convert S21 to dB scale.

        Returns:
        - dict: A dictionary {label: S21 values (np.ndarray)}.
        """
        s_params_data = {}
        for label, network in self.ntwk_dict.items():
            s21 = network.s[:, 1, 0]
            if db:
                s21 = 20 * np.log10(np.abs(s21))
            s_params_data[label] = s21
        return s_params_data
    
    def plot_impedance(self, title='Impedance Measurement', ymax=None, ymin=None, 
                       plot_imaginary=True,
                       save_path=None) -> None:
        """
        Plot multiple impedances from .s1p Network objects on the same axes.

        Parameters:
        - title (str): Title of the plot.
        - ymax (float): Maximum y-axis limit.
        - ymin (float): Minimum y-axis limit.
        - save_path (str): filepath to save the combined plot

        Returns:
        - dict: the same ntwk_dict passed in
        """
        ntwk_dict = self.ntwk_dict
        if not ntwk_dict:
            print("No networks provided.")
            return ntwk_dict

        fig, ax = plt.subplots(figsize=(14, 8))
        if plot_imaginary:
            # Replace single axis with two stacked axes
            fig.clf()
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
            ax = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax)
        else:
            ax2 = None

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, (label, network) in enumerate(ntwk_dict.items()):
            freq = network.f
            z_real = network.z[:, 0, 0].real
            z_imag = network.z[:, 0, 0].imag
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(freq / 1e6, z_real, label=f'{label}', color=color)
            if plot_imaginary:
                ax2.plot(freq / 1e6, z_imag, color=color, linestyle='--', label=f'{label}')

        if not plot_imaginary:
            ax.set_xlabel('Frequency [MHz]', fontsize=20)
        else:
            ax2.set_xlabel('Frequency [MHz]', fontsize=20)

        ax.set_ylabel('Real Impedance [Ω]', fontsize=20)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)

        ax.legend(loc='best', fontsize=18)
        ax.set_title(title, fontsize=22)

        if plot_imaginary:
            ax2.set_ylabel('Imaginary Impedance [Ω]', fontsize=20)
            ax2.grid(True)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.legend(loc='best', fontsize=18)

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_reflection_loss(self, db=True, title='Reflection Measurement (S11)', ymax=None, ymin=None, 
                             show_phase=False, attenuation=0, save_path=None):
        """
        Plot multiple reflections from .s1p Network objects on the same axes.

        Parameters:
        - db (bool): If True, plot reflection in dB
        - show_phase (bool): If True, also plot phase in degrees (dashed lines)
        - attenuation (float): Attenuation added to the magnitude (dB)
        - save_path (str): filepath to save the combined plot

        Returns:
        - dict: the same ntwk_dict passed in
        """
        ntwk_dict = self.ntwk_dict
        if not ntwk_dict:
            print("No networks provided.")
            return ntwk_dict

        fig, ax1 = plt.subplots(figsize=(14, 8))
        if show_phase:
            # replace the single axis with two stacked axes (top: magnitude, bottom: phase)
            fig.clf()
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
        else:
            ax2 = None

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, (label, network) in enumerate(ntwk_dict.items()):
            freq = network.f
            s11 = network.s[:, 0, 0]
            mag = 20 * np.log10(np.abs(s11)) + attenuation if db else np.abs(s11)
            phase = np.angle(s11, deg=True)

            color = color_cycle[idx % len(color_cycle)]
            ax1.plot(freq / 1e6, mag, label=f'{label}', color=color)
            if show_phase:
                ax2.plot(freq / 1e6, phase, color=color, linestyle='--', label=f'{label} (phase)')

        if not show_phase:
            ax1.set_xlabel('Frequency [MHz]', fontsize=20)
        else:
            ax2.set_xlabel('Frequency [MHz]', fontsize=20)

        ax1.set_ylabel('Reflection' + (' [dB]' if db else ''), fontsize=20)
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        if ymax is not None:
            ax1.set_ylim(top=ymax)
        if ymin is not None:
            ax1.set_ylim(bottom=ymin)

        if show_phase:
            ax2.set_ylabel('Phase [deg]', fontsize=18)
            ax2.tick_params(axis='y', labelsize=16)
            ax2.grid(True)

            # Separate legends for magnitude and phase
            h1, l1 = ax1.get_legend_handles_labels()
            if h1:
                ax1.legend(h1, l1, fontsize=18, loc='best')
            h2, l2 = ax2.get_legend_handles_labels()
            if h2:
                ax2.legend(h2, l2, fontsize=18, loc='best')
        else:
            ax1.legend(loc='best', fontsize=18)

        ax1.set_title(title, fontsize=22)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
    
    def plot_smith_chart(self, save_path=None, save_plot=True, title='Smith Chart',
                        freq_range=None, marker_freq=None, autoscale=False):
        """
        Plot Smith chart from one or more scikit-rf Network objects.

        Parameters:
        - suffix (str): Used for output filename if saving.
        - freq_range (tuple): (min_freq, max_freq) in Hz to restrict plotting range. 
                            If None, plots all frequencies.
        - marker_freq (list): List of frequencies in Hz to mark on the Smith chart.
        """
        ntwk_dict = self.ntwk_dict.copy()
        # Filter networks by frequency range if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            filtered_ntwk_dict = {}
            for label, ntwk in ntwk_dict.items():
                # Create frequency mask
                freq_mask = (ntwk.f >= min_freq) & (ntwk.f <= max_freq)
                if np.any(freq_mask):
                    # Create new network with filtered frequencies
                    filtered_ntwk = rf.Network(f=ntwk.f[freq_mask], s=ntwk.s[freq_mask], z0=ntwk.z0[freq_mask])
                    filtered_ntwk_dict[label] = filtered_ntwk
                else:
                    print(f"Warning: No frequencies in range for {label}")
            ntwk_dict = filtered_ntwk_dict
        
        if not ntwk_dict:
            print("No networks to plot after frequency filtering")
            return
        
        fig, ax = plt.subplots()
        fig.set_size_inches(14, 12)
        for label, ntwk in ntwk_dict.items():
            # Extract only S11 for Smith chart plotting
            s11_ntwk = rf.Network(f=ntwk.f, s=ntwk.s[:, 0:1, 0:1], z0=ntwk.z0)
            s11_ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)

        for text in ax.texts:
            text.set_fontsize(18)
        
        if autoscale:
            ax.autoscale()
            ax.set_xlim(1.0)

        # Update axis labels (Real and Imaginary)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, labelpad=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, labelpad=16)

        ax.set_title(title, fontsize=20)
        
        if marker_freq is not None:
            for mfreq in marker_freq:
                for label, ntwk in ntwk_dict.items():
                    # Find closest frequency index
                    idx = (np.abs(ntwk.f - mfreq)).argmin()
                    s11_point = ntwk.s[idx, 0, 0]
                    impedance = ntwk.z[idx, 0, 0]
                    impedance = f'{impedance.real:.1f} + j{impedance.imag:.1f} Ω'
                    if len(ntwk_dict) > 1:
                        label = f'{label} @ {mfreq/1e6:.2f} MHz: {impedance}' 
                    else:
                        label = f'{mfreq/1e6:.2f} MHz: {impedance}'
                    ax.plot(np.real(s11_point), np.imag(s11_point), 'o', markersize=7, label=label)
        
        ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
        plt.tight_layout()
        
        if save_plot:
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_gain(self, attenuation=0, title='Gain Measurement', y_range=(None, None),
                  marker_freqs=None, save_path=None, smoothing=False, smoothing_kwargs=None):
        """
        Plot gain from multiple .s2p Network objects on the same axes.

        Parameters:
        - title (str): Title of the plot.
        - attenuation (float): Attenuation used during measurement (dB). Plotted gain would be (measured gain + attenuation).
        - y_range (tuple): (y_min, y_max) limits for the y-axis. Use (None, None) for auto-scaling.
        - save_path (str): full path to save the plot
        - marker_freqs (list): List of frequencies in MHz to mark on the gain plot.
        - smoothing (bool): Whether to apply smoothing to the gain curves.
        - smoothing_kwargs (dict): Additional keyword arguments for smoothing function.

        Returns:
        - dict: the same ntwk_dict passed in
        """
        freq = self.get_freq(MHz=True)
        gain = self.get_s21(db=True)
        if smoothing:
            from highz_exp.spec_proc import smooth_spectrum
            smoothing_kwargs = smoothing_kwargs or {}
            gain_smoothed = {}
            for label in gain:
                gain_smoothed[label] = smooth_spectrum(gain[label], **smoothing_kwargs)
                # Plot both raw and smoothed
                ordered_labels = list(self.ntwk_dict.keys())
                ordered_gains = [gain[label] + attenuation for label in ordered_labels]
                ordered_gains_smoothed = [gain_smoothed[label] + attenuation for label in ordered_labels]
                ordered_labels_combined = [f'{label} (raw)' for label in ordered_labels] + [f'{label} (smoothed)' for label in ordered_labels]
                ordered_gains_combined = ordered_gains + ordered_gains_smoothed
                plot_gain(freq, ordered_gains_combined, label=ordered_labels_combined, title=title, y_range=y_range, 
                    save_path=save_path, marker_freqs=marker_freqs)
        else:
            ordered_labels = list(self.ntwk_dict.keys())
            ordered_gains = [gain[label] + attenuation for label in ordered_labels]
            plot_gain(freq, ordered_gains, label=ordered_labels, title=title, y_range=y_range, 
                    save_path=save_path,
                    marker_freqs=marker_freqs)
    
    def apply_to_all_s11s(self, func, inplace=True):
        """Apply a function to the S11 parameters of all networks."""
        new_ntwk_dict = {}
        for key, value in self.ntwk_dict.items():
            ntwk_copy = copy.deepcopy(self.ntwk_dict[key])
            spectrum = ntwk_copy.s[:, 0, 0]
            updated_spectrum = func(spectrum)
            ntwk_copy.s[:, 0, 0] = updated_spectrum
            new_ntwk_dict[key] = ntwk_copy
        if inplace:
            self.ntwk_dict = new_ntwk_dict
        return new_ntwk_dict
    
    def keep_freq(self, freq_min, freq_max, inplace=True) -> dict:
        """Keep only frequencies within [freq_min, freq_max] for all networks.
        
        Parameters:
        - freq_min (float): Minimum frequency to keep (Hz).
        - freq_max (float): Maximum frequency to keep (Hz).
        - inplace (bool): If True, modify the ntwk_dict in place.
        
        Return 
        - dict: New dictionary of networks with frequencies outside the range removed."""
        new_ntwk_dict = {}
        for key, value in self.ntwk_dict.items():
            ntwk_copy = copy.deepcopy(self.ntwk_dict[key])
            indices_to_keep = np.where((ntwk_copy.f >= freq_min) & (ntwk_copy.f <= freq_max))[0]
            new_ntwk = ntwk_copy[indices_to_keep]
            new_ntwk_dict[key] = new_ntwk
        if inplace:
            self.ntwk_dict = new_ntwk_dict
        return new_ntwk_dict

    def interpolate_all(self, new_freqs, inplace=True):
        """Interpolate all networks to a new frequency axis.
        
        Parameters:
        - new_freqs (np.ndarray): New frequency axis (Hz).
        - inplace (bool): If True, modify the ntwk_dict in place. """
        new_ntwk_dict = {}
        for key, value in self.ntwk_dict.items():
            ntwk_copy = copy.deepcopy(self.ntwk_dict[key])
            ntwk_interp = ntwk_copy.interpolate(new_freqs)
            new_ntwk_dict[key] = ntwk_interp
        if inplace:
            self.ntwk_dict = new_ntwk_dict
        return new_ntwk_dict

    
    @staticmethod
    def subtract_s11_networks(ntwk1, ntwk2, new_name=None):
        """
        Create a new network where S11 = ntwk1.S11 - ntwk2.S11

        Parameters:
        - ntwk1 (skrf.Network): First network
        - ntwk2 (skrf.Network): Second network to subtract
        - new_name (str, optional): Name for the new network. If None, auto-generates name.

        Returns:
        - skrf.Network: New network with S11 = ntwk1.S11 - ntwk2.S11
        """
        import copy

        # Verify frequencies match
        if not np.allclose(ntwk1.f, ntwk2.f):
            raise ValueError("Network frequencies don't match!")

        # Verify both networks have the same dimensions
        if ntwk1.s.shape != ntwk2.s.shape:
            raise ValueError("Network S-parameter dimensions don't match!")

        # Create new network by copying the first one
        new_ntwk = copy.deepcopy(ntwk1)

        # Subtract S11 parameters
        new_ntwk.s[:, 0, 0] = ntwk1.s[:, 0, 0] - ntwk2.s[:, 0, 0]

        # Set the name
        if new_name is None:
            name1 = getattr(ntwk1, 'name', 'ntwk1')
            name2 = getattr(ntwk2, 'name', 'ntwk2')
            new_ntwk.name = f"{name1}_minus_{name2}"
        else:
            new_ntwk.name = new_name

        return new_ntwk
            
def k_factor(s_params):
    """
    Compute Rollet's stability factor (K) for an array of 2-port S-parameters.

    Parameters
    ----------
    s_params : ndarray of shape (n_freqs, 2, 2)
        Complex S-parameter matrices at each frequency.

    Returns
    -------
    k : ndarray of shape (n_freqs,)
        Rollet's stability factor at each frequency.
    delta : ndarray of shape (n_freqs,)
        Determinant of the S-parameter matrix at each frequency.
    """
    s11 = s_params[:, 0, 0]
    s12 = s_params[:, 0, 1]
    s21 = s_params[:, 1, 0]
    s22 = s_params[:, 1, 1]

    delta = s11 * s22 - s12 * s21
    numerator = 1 - np.abs(s11)**2 - np.abs(s22)**2 + np.abs(delta)**2
    denominator = 2 * np.abs(s12 * s21)

    k = numerator / denominator
    return k, delta


def interpolate_ntwk_dict(ntwk_dict, target_freqs, freq_range=None) -> dict:
    """
    Interpolate all ntwk objects in a dictionary to the target frequencies and remove frequencies outside the specified range.

    Parameters:
    - ntwk_dict (dict): Dictionary of {'label': skrf.Network}
    - target_freqs (array-like): Frequencies to interpolate to (in Hz)
    - freq_range (tuple, optional): (min_freq, max_freq) to override common range

    Returns:
    - dict: New dictionary with deepcopied and interpolated skrf.Network objects
            with frequencies outside freq_range removed
    """
    # Find the common frequency range across all networks
    common_min_freq = max(np.min(ntwk.f) for ntwk in ntwk_dict.values())
    common_max_freq = min(np.max(ntwk.f) for ntwk in ntwk_dict.values())

    # Determine the frequency range to use
    if freq_range is not None:
        min_freq, max_freq = freq_range
        if not (common_min_freq <= min_freq <= max_freq <= common_max_freq):
            print(f"Warning: Requested range {freq_range[0]/1e6:.1f}-{freq_range[1]/1e6:.1f} MHz is outside common range "
                  f"{common_min_freq/1e6:.1f}-{common_max_freq/1e6:.1f} MHz")
            # Clip the requested range to the available data range
            min_freq = max(min_freq, common_min_freq)
            max_freq = min(max_freq, common_max_freq)
    else:
        min_freq, max_freq = common_min_freq, common_max_freq

    new_ntwk_dict = {}
    for label, ntwk in ntwk_dict.items():
        # Get the actual frequency range for this specific network
        ntwk_min_freq = np.min(ntwk.f)
        ntwk_max_freq = np.max(ntwk.f)

        # Clip target frequencies to both the desired range AND the network's actual range
        effective_min = max(min_freq, ntwk_min_freq)
        effective_max = min(max_freq, ntwk_max_freq)

        # Filter target frequencies to the effective range
        mask = (target_freqs >= effective_min) & (target_freqs <= effective_max)
        clipped_freqs = target_freqs[mask]

        if len(clipped_freqs) == 0:
            print(f"Warning: No target frequencies within valid range for {label}")
            continue

        # Create a copy and interpolate to the clipped frequencies
        ntwk_copy = copy.deepcopy(ntwk)
        interp_ntwk = ntwk_copy.interpolate(clipped_freqs)
        new_ntwk_dict[label] = interp_ntwk

    return new_ntwk_dict