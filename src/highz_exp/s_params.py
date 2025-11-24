import numpy as np
import skrf as rf
import pickle
import os
from matplotlib import pyplot as plt

pjoin = os.path.join

class S_Params:
    def __init__(self, s_params_files=None, labels=None, ntwk_dict=None, pickle_file=None):
        """
        Construct S_Params in one of three ways:

        - Provide ntwk_dict: a dict of {label: rf.Network} or {label: filepath}
        - Provide pickle_file: path to a pickled dict of networks (same format as ntwk_dict)
        - Provide s_params_files (str or list) and optional labels (str or list)
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
    
    def get_s21(self, db=True):
        """
        Load S-parameters from .s2p files in ntwk_dict, converting to dB if specified.

        Parameters:
        - db (bool): If True, convert S11 to dB scale.

        Returns:
        - dict: A dictionary with the same keys as ntwk_dict and values as S11 arrays.
        """
        s_params_data = {}
        for label, network in self.ntwk_dict.items():
            s21 = network.s[:, 1, 0]
            if db:
                s21 = 20 * np.log10(np.abs(s21))
            s_params_data[label] = s21
        return s_params_data

    def plot_s1p(self, db=True, title='Reflection Measurement (S11)', ymax=None, ymin=None, show_phase=False, attenuation=0, save_dir=None, suffix=None):
        """
        Plot multiple reflections from .s1p Network objects on the same axes.

        Parameters:
        - db (bool): If True, plot reflection in dB
        - show_phase (bool): If True, also plot phase in degrees (dashed lines)
        - attenuation (float): Attenuation added to the magnitude (dB)
        - save_dir (str): directory to save the combined plot
        - suffix (str): optional suffix for saved filename

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

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            safe_suffix = f"_{suffix}" if suffix else ""
            plt.savefig(f'{save_dir}/Reflection{safe_suffix}.png', dpi=150, bbox_inches='tight')

        plt.show()
    
        
    def plot_smith_chart(self, suffix='LNA', save_plot=True, save_dir=None, title='Smith Chart',
                        freq_range=None):
        """
        Plot Smith chart from one or more scikit-rf Network objects.
        Parameters:
        - suffix (str): Used for output filename if saving.
        - freq_range (tuple): (min_freq, max_freq) in Hz to restrict plotting range. 
                            If None, plots all frequencies.
        """
        ntwk_dict = self.ntwk_dict
        # Filter networks by frequency range if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            filtered_ntwk_dict = {}
            for label, ntwk in ntwk_dict.items():
                # Create frequency mask
                freq_mask = (ntwk.f >= min_freq) & (ntwk.f <= max_freq)
                if np.any(freq_mask):
                    # Create new network with filtered frequencies
                    filtered_ntwk = ntwk.copy()
                    filtered_ntwk.f = ntwk.f[freq_mask]
                    filtered_ntwk.s = ntwk.s[freq_mask]
                    filtered_ntwk_dict[label] = filtered_ntwk
                else:
                    print(f"Warning: No frequencies in range for {label}")
            ntwk_dict = filtered_ntwk_dict
        
        if not ntwk_dict:
            print("No networks to plot after frequency filtering")
            return
        
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        for label, ntwk in ntwk_dict.items():
            ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)
        
        for text in ax.texts:
            text.set_fontsize(18)

        # Update axis labels (Real and Imaginary)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, labelpad=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, labelpad=18)

        ax.set_title(title, fontsize=20)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0, fontsize=18)
        plt.tight_layout()
        
        if save_plot:
            suffix = suffix.replace(' ', '_')
            # Save to current directory if no path info is available
            if save_dir is None:
                save_dir = os.getcwd()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Add frequency range to filename if specified
            filename = f'{suffix}_smith_chart'
            filename += '.png'
            
            fig.savefig(pjoin(save_dir, filename), bbox_inches='tight')
        plt.show()
            
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