#!/usr/bin/env python3
"""
Interactive inspector for state spectra across cycles.
View all spectra from a specific state in each cycle, navigate between cycles with arrow keys.
"""

import argparse
import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Optional
import sys

# Import functions from waterfall_plotter
from waterfall_plotter import (
    load_state_file
)


class SpectrumInspector:
    """Interactive viewer for state spectra across cycles."""
    
    def __init__(self, day_dir: Path, state: str, filter_num: int, reference_spec: Optional[str] = None):
        self.day_dir = day_dir
        self.state = state
        self.filter_num = filter_num
        self.current_cycle_idx = 0
        
        # Expected number of LO points (650-934 MHz in 2 MHz steps)
        self.expected_n_points = 143  # Should be (934-650)/2 + 1 = 143
        
        # Find all cycles with this state
        self.cycle_dirs = self._find_cycles()
        
        # Load global reference spectrum if specified
        self.reference_rf_freqs = None
        self.reference_powers = None
        self.reference_info = None
        
        if reference_spec:
            self._load_reference_spectrum(reference_spec)
        
        if not self.cycle_dirs:
            print(f"Error: No cycles found with state_{state}.fits")
            sys.exit(1)
        
        print(f"Found {len(self.cycle_dirs)} cycles with state_{state}.fits")
        
        # Set up the plot
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.15)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous (←)')
        self.btn_next = Button(ax_next, 'Next (→)')
        
        self.btn_prev.on_clicked(self.prev_cycle)
        self.btn_next.on_clicked(self.next_cycle)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Display first cycle
        self.update_plot()
        
    def _find_cycles(self):
        """Find all cycle directories containing the requested state."""
        cycle_dirs = []
        for d in sorted(self.day_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith('cycle_'):
                continue
            
            state_file = d / f"state_{self.state}.fits"
            state_file_alt = d / f"state_{self.state}_OC.fits"
            
            if state_file.exists() or state_file_alt.exists():
                cycle_dirs.append(d)
        
        return cycle_dirs
    
    def _load_reference_spectrum(self, reference_spec: str):
        """
        Load a global reference spectrum.
        
        Args:
            reference_spec: Format "cycle_name:index" e.g., "cycle_010_11022025_021101:1"
                           or just "cycle_010:1" (will match partial name)
        """
        try:
            parts = reference_spec.split(':')
            if len(parts) != 2:
                print(f"Error: Reference format should be 'cycle_name:index' (e.g., 'cycle_010:1')")
                return
            
            cycle_pattern, idx_str = parts
            spectrum_idx = int(idx_str)
            
            # Find matching cycle directory
            matching_cycles = [d for d in self.cycle_dirs if cycle_pattern in d.name]
            
            if not matching_cycles:
                print(f"Error: No cycle matching '{cycle_pattern}' found")
                print(f"Available cycles: {[d.name for d in self.cycle_dirs[:5]]}...")
                return
            
            if len(matching_cycles) > 1:
                print(f"Warning: Multiple cycles match '{cycle_pattern}', using first: {matching_cycles[0].name}")
            
            ref_cycle_dir = matching_cycles[0]
            
            # Load the cycle data
            waterfall, _, state_file = self._load_cycle_data(ref_cycle_dir)
            
            if spectrum_idx >= len(waterfall.timestamps):
                print(f"Error: Spectrum index {spectrum_idx} out of range (cycle has {len(waterfall.timestamps)} spectra)")
                return
            
            # Store reference
            self.reference_rf_freqs = waterfall.rf_frequencies[spectrum_idx]
            self.reference_powers = waterfall.powers[spectrum_idx]
            self.reference_info = {
                'cycle': ref_cycle_dir.name,
                'index': spectrum_idx,
                'timestamp': waterfall.timestamps[spectrum_idx]
            }
            
            print(f"\n{'='*70}")
            print(f"LOADED REFERENCE SPECTRUM:")
            print(f"  Cycle: {self.reference_info['cycle']}")
            print(f"  Index: {self.reference_info['index']}")
            print(f"  Timestamp: {self.reference_info['timestamp']}")
            print(f"  Mean power: {np.mean(self.reference_powers):.1f} dBm")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"Error loading reference spectrum: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_cycles(self):
        """Find all cycle directories containing the requested state."""
        cycle_dirs = []
        for d in sorted(self.day_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith('cycle_'):
                continue
            
            state_file = d / f"state_{self.state}.fits"
            state_file_alt = d / f"state_{self.state}_OC.fits"
            
            if state_file.exists() or state_file_alt.exists():
                cycle_dirs.append(d)
        
        return cycle_dirs
    
    def _load_cycle_data(self, cycle_dir: Path):
        """Load all spectra from this cycle's state file."""
        state_file = cycle_dir / f"state_{self.state}.fits"
        if not state_file.exists():
            state_file = cycle_dir / f"state_{self.state}_OC.fits"
        
        # Load using waterfall_plotter function
        waterfall, metadata = load_state_file(state_file, cycle_dir=cycle_dir, 
                                             filter_num=self.filter_num, verbose=False)
        
        return waterfall, metadata, state_file
    
    def _compare_to_reference(self, rf_freqs: np.ndarray, powers: np.ndarray,
                             ref_rf_freqs: np.ndarray, ref_powers: np.ndarray) -> dict:
        """
        Compare a spectrum to a reference (known-good) spectrum.
        
        Returns dict with:
            - is_suspect: bool - True if too different from reference
            - reasons: list[str] - Why it's suspect
            - metrics: dict - Comparison metrics
        """
        from scipy.interpolate import interp1d
        
        reasons = []
        
        # Sort both spectra by RF frequency
        sort_idx = np.argsort(rf_freqs)
        rf_sorted = rf_freqs[sort_idx]
        power_sorted = powers[sort_idx]
        
        ref_sort_idx = np.argsort(ref_rf_freqs)
        ref_rf_sorted = ref_rf_freqs[ref_sort_idx]
        ref_power_sorted = ref_powers[ref_sort_idx]
        
        # Interpolate test spectrum onto reference frequency grid
        try:
            interp_func = interp1d(rf_sorted, power_sorted, 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
            power_interp = interp_func(ref_rf_sorted)
            
            # Calculate metrics where both have valid data
            valid_mask = ~np.isnan(power_interp)
            if np.sum(valid_mask) < 10:
                return {'is_suspect': False, 'reasons': ['Not enough overlap'], 'metrics': {}}
            
            # Calculate difference
            diff = power_interp[valid_mask] - ref_power_sorted[valid_mask]
            
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            max_diff = np.max(np.abs(diff))
            
            # Calculate correlation
            correlation = np.corrcoef(power_interp[valid_mask], ref_power_sorted[valid_mask])[0, 1]
            
            metrics = {
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'max_diff': max_diff,
                'correlation': correlation,
                'overlap_points': np.sum(valid_mask)
            }
            
            # Detection criteria
            # 1. Large systematic offset (more than 10 dB difference on average)
            if abs(mean_diff) > 5.5:  # Slightly tightened from 6.0
                reasons.append(f'Large offset from reference (mean Δ={mean_diff:.1f} dB)')
            
            # 2. High variability in difference (spectra have different shape)
            if std_diff > 2.8:  # Slightly tightened from 3.0
                reasons.append(f'Different shape from reference (σ={std_diff:.1f} dB)')
            
            # 3. Poor correlation (< 0.82 means very different pattern)
            if correlation < 0.82:  # Slightly tightened from 0.80
                reasons.append(f'Low correlation with reference (r={correlation:.3f})')
            
            return {
                'is_suspect': len(reasons) > 0,
                'reasons': reasons,
                'metrics': metrics
            }
            
        except Exception as e:
            return {'is_suspect': False, 'reasons': [f'Comparison failed: {e}'], 'metrics': {}}
    
    def _check_spectrum_quality(self, rf_freqs: np.ndarray, powers: np.ndarray) -> dict:
        """
        Check if a spectrum appears to be shifted due to LO/ADC sync issues.
        
        The issue: ADC starts collecting before LO sweep starts, so the
        beginning shows flat noise floor around -60 dBm for ~first third of spectrum.
        
        Detection: Look for flat, low-power region at the beginning.
        
        Returns dict with:
            - is_suspect: bool - True if spectrum appears problematic
            - reasons: list[str] - List of reasons why spectrum is suspect
            - edge_stats: dict - Statistics about spectrum edges
        """
        reasons = []
        
        # Sort by RF frequency to analyze spectrum from low to high freq
        sort_idx = np.argsort(rf_freqs)
        rf_sorted = rf_freqs[sort_idx]
        power_sorted = powers[sort_idx]
        
        n_points = len(power_sorted)
        if n_points < 20:  # Need enough points to analyze
            return {'is_suspect': False, 'reasons': [], 'edge_stats': {}}
        
        # Analyze first third of spectrum (where noise floor appears in bad spectra)
        first_third_size = max(15, int(n_points * 0.33))
        
        first_third_power = power_sorted[:first_third_size]
        rest_power = power_sorted[first_third_size:]
        
        # Calculate statistics
        first_third_mean = np.mean(first_third_power)
        first_third_std = np.std(first_third_power)
        rest_mean = np.mean(rest_power)
        rest_std = np.std(rest_power)
        
        edge_stats = {
            'begin_mean': first_third_mean,
            'begin_std': first_third_std,
            'middle_mean': rest_mean,
            'middle_std': rest_std,
            'end_mean': rest_mean,  # For compatibility
            'end_std': rest_std,
            'begin_gradient': 0.0,
            'end_gradient': 0.0,
        }
        
        # Detection: First third is FLAT (low std) AND low power (around -60 dBm or below -50)
        # AND significantly different from rest of spectrum
        
        is_flat_noise = first_third_std < 3.0  # Low variation (< 3 dB std)
        is_low_power = first_third_mean < -45.0  # Lower threshold to catch more cases
        is_different_from_rest = (rest_mean - first_third_mean) > 5.0  # At least 5 dB difference
        
        if is_flat_noise and is_low_power and is_different_from_rest:
            reasons.append(f'Noise floor at start: {first_third_mean:.1f} dBm (σ={first_third_std:.2f}), '
                          f'{rest_mean - first_third_mean:.1f} dB below rest')
        
        # Also print to console for debugging
        if first_third_mean < -45:
            print(f"    [Checking] First 1/3: {first_third_mean:.1f}±{first_third_std:.2f} dBm, "
                  f"Rest: {rest_mean:.1f}±{rest_std:.2f} dBm, "
                  f"Diff: {rest_mean - first_third_mean:.1f} dB, "
                  f"Flags: flat={is_flat_noise}, low={is_low_power}, diff={is_different_from_rest}")
        
        return {
            'is_suspect': len(reasons) > 0,
            'reasons': reasons,
            'edge_stats': edge_stats
        }
    
    def update_plot(self):
        """Update the plot with current cycle data."""
        cycle_dir = self.cycle_dirs[self.current_cycle_idx]
        waterfall, metadata, state_file = self._load_cycle_data(cycle_dir)
        
        n_spectra = len(waterfall.timestamps)
        filter_center = 904.0 + 2.6 * self.filter_num
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Check quality of all spectra (no longer excluding first/last by position)
        excluded_indices = set()  # No automatic exclusion
        suspect_indices = set()
        quality_info = {}
        
        # Determine if this is a calibration state (apply quality checks)
        # Calibration states: 2, 3, 4, 5, 6, 7, 1_OC, 0
        # State 1 is NOT a calibration state (skip quality checks)
        is_calibration_state = self.state not in ['1']
        
        # If reference spectrum specified AND this is a calibration state, use relative comparison
        if self.reference_rf_freqs is not None and is_calibration_state:
            print(f"\n[Comparing to global reference: {self.reference_info['cycle']}, spectrum {self.reference_info['index']}]")
            
            for i in range(n_spectra):
                rf_freqs = waterfall.rf_frequencies[i]
                powers = waterfall.powers[i]
                quality = self._compare_to_reference(rf_freqs, powers, 
                                                     self.reference_rf_freqs, self.reference_powers)
                quality_info[i] = quality
                
                if quality['is_suspect']:
                    suspect_indices.add(i)
            
            # Print summary of comparison metrics
            all_correlations = [q['metrics'].get('correlation', 1.0) for q in quality_info.values() if 'metrics' in q and q['metrics']]
            all_mean_diffs = [q['metrics'].get('mean_diff', 0.0) for q in quality_info.values() if 'metrics' in q and q['metrics']]
            
            if all_correlations:
                print(f"[Correlation with reference: min={min(all_correlations):.3f}, max={max(all_correlations):.3f}, mean={np.mean(all_correlations):.3f}]")
            if all_mean_diffs:
                print(f"[Mean offset from reference: min={min(all_mean_diffs):.1f}, max={max(all_mean_diffs):.1f}, mean={np.mean(all_mean_diffs):.1f} dB]")
        
        else:
            # Use absolute detection (original method)
            # First pass: collect all statistics to understand the data
            all_begin_stds = []
            all_middle_stds = []
            all_power_diffs = []
            all_gradients = []
            
            for i in range(n_spectra):
                rf_freqs = waterfall.rf_frequencies[i]
                powers = waterfall.powers[i]
                quality = self._check_spectrum_quality(rf_freqs, powers)
                quality_info[i] = quality
                
                if quality['edge_stats']:
                    stats = quality['edge_stats']
                    all_begin_stds.append(stats['begin_std'])
                    all_middle_stds.append(stats['middle_std'])
                    all_power_diffs.append(stats['middle_mean'] - stats['begin_mean'])
                    all_gradients.append(stats['begin_gradient'])
            
            # Print statistics summary to help tune thresholds
            if all_begin_stds:
                print(f"\n[DEBUG] Statistics across all {n_spectra} spectra:")
                print(f"  Begin std: min={min(all_begin_stds):.2f}, max={max(all_begin_stds):.2f}, mean={np.mean(all_begin_stds):.2f}")
                print(f"  Middle std: min={min(all_middle_stds):.2f}, max={max(all_middle_stds):.2f}, mean={np.mean(all_middle_stds):.2f}")
                print(f"  Power diff (mid-begin): min={min(all_power_diffs):.2f}, max={max(all_power_diffs):.2f}, mean={np.mean(all_power_diffs):.2f}")
                print(f"  Begin gradient: min={min(all_gradients):.2f}, max={max(all_gradients):.2f}, mean={np.mean(all_gradients):.2f}")
            
            # Second pass: mark suspects based on detection criteria
            for i, quality in quality_info.items():
                if quality['is_suspect']:
                    suspect_indices.add(i)
        
        # Plot configuration
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_spectra, 10)))
        
        # Top plot: All spectra overlaid
        ax_all = self.axes[0]
        for i in range(n_spectra):
            rf_freqs = waterfall.rf_frequencies[i]
            powers = waterfall.powers[i]
            timestamp = waterfall.timestamps[i]
            
            # Sort by frequency
            sort_idx = np.argsort(rf_freqs)
            rf_freqs_sorted = rf_freqs[sort_idx]
            powers_sorted = powers[sort_idx]
            
            # Determine style based on quality
            quality_markers = []
            if i in excluded_indices:
                quality_markers.append('EXCL')
            if i in suspect_indices:
                quality_markers.append('SUSPECT')
            
            if quality_markers:
                linestyle = '--'
                alpha = 0.4
                linewidth = 1.0
                label = f'{i}: {timestamp} [{"/".join(quality_markers)}]'
            else:
                linestyle = '-'
                alpha = 0.8
                linewidth = 1.5
                label = f'{i}: {timestamp}'
            
            ax_all.plot(rf_freqs_sorted, powers_sorted, 
                       color=colors[i], linestyle=linestyle, 
                       alpha=alpha, linewidth=linewidth, label=label)
        
        ax_all.set_xlabel('RF Frequency (MHz)', fontsize=11)
        ax_all.set_ylabel('Power (dBm)', fontsize=11)
        ax_all.set_title(f'{cycle_dir.name} - State {self.state} - Filter {self.filter_num}\n'
                        f'All {n_spectra} Spectra (dashed = suspect sync issues)', fontsize=12, fontweight='bold')
        ax_all.grid(True, alpha=0.3)
        ax_all.legend(loc='upper right', fontsize=8, ncol=2)
        
        # Middle plot: Only GOOD spectra (non-suspect, what should go into waterfall)
        ax_included = self.axes[1]
        included_count = 0
        for i in range(n_spectra):
            if i in suspect_indices:
                continue
            
            rf_freqs = waterfall.rf_frequencies[i]
            powers = waterfall.powers[i]
            timestamp = waterfall.timestamps[i]
            
            sort_idx = np.argsort(rf_freqs)
            rf_freqs_sorted = rf_freqs[sort_idx]
            powers_sorted = powers[sort_idx]
            
            ax_included.plot(rf_freqs_sorted, powers_sorted,
                           color=colors[i], linewidth=1.5, 
                           label=f'{i}: {timestamp}')
            included_count += 1
        
        ax_included.set_xlabel('RF Frequency (MHz)', fontsize=11)
        ax_included.set_ylabel('Power (dBm)', fontsize=11)
        ax_included.set_title(f'GOOD Spectra Only ({included_count} spectra, no sync issues detected)', 
                            fontsize=12)
        ax_included.grid(True, alpha=0.3)
        if included_count > 0:
            ax_included.legend(loc='upper right', fontsize=8, ncol=2)
        
        # Bottom plot: Only SUSPECT spectra (detected sync issues)
        ax_excluded = self.axes[2]
        excluded_count = 0
        for i in suspect_indices:
            rf_freqs = waterfall.rf_frequencies[i]
            powers = waterfall.powers[i]
            timestamp = waterfall.timestamps[i]
            
            sort_idx = np.argsort(rf_freqs)
            rf_freqs_sorted = rf_freqs[sort_idx]
            powers_sorted = powers[sort_idx]
            
            ax_excluded.plot(rf_freqs_sorted, powers_sorted,
                           color=colors[i], linewidth=2.0,
                           label=f'{i}: {timestamp}')
            excluded_count += 1
        
        ax_excluded.set_xlabel('RF Frequency (MHz)', fontsize=11)
        ax_excluded.set_ylabel('Power (dBm)', fontsize=11)
        ax_excluded.set_title(f'SUSPECT Spectra (sync issues, {excluded_count} spectra)', 
                            fontsize=12, color='red')
        ax_excluded.grid(True, alpha=0.3)
        if excluded_count > 0:
            ax_excluded.legend(loc='upper right', fontsize=8)
        
        # Update window title
        self.fig.suptitle(f'Cycle {self.current_cycle_idx + 1}/{len(self.cycle_dirs)} - '
                         f'Use ← → arrow keys to navigate',
                         fontsize=14, fontweight='bold')
        
        # Print info to console
        print(f"\n{'='*70}")
        print(f"Cycle {self.current_cycle_idx + 1}/{len(self.cycle_dirs)}: {cycle_dir.name}")
        print(f"State file: {state_file.name}")
        print(f"Total spectra: {n_spectra}")
        print(f"Good spectra: {n_spectra - len(suspect_indices)}")
        print(f"Suspect (sync issues): {len(suspect_indices)}")
        
        # Print details of suspect spectra
        if suspect_indices:
            print(f"\n{'='*70}")
            print("SUSPECT SPECTRA (different from global reference):" if self.reference_rf_freqs is not None
                  else "SUSPECT SPECTRA (likely LO/ADC sync issues):")
            for i in sorted(suspect_indices):
                q = quality_info[i]
                print(f"\n  Spectrum {i}: {waterfall.timestamps[i]}")
                
                if 'metrics' in q and q['metrics']:
                    m = q['metrics']
                    print(f"    Comparison to reference:")
                    print(f"      Mean offset: {m.get('mean_diff', 0):.1f} dBm")
                    print(f"      Variability: σ={m.get('std_diff', 0):.2f} dBm")
                    print(f"      Correlation: r={m.get('correlation', 0):.3f}")
                    print(f"      Max difference: {m.get('max_diff', 0):.1f} dBm")
                elif 'edge_stats' in q and q['edge_stats']:
                    stats = q['edge_stats']
                    print(f"    Edge Analysis:")
                    print(f"      Start: mean={stats['begin_mean']:.1f} dBm, σ={stats['begin_std']:.2f}")
                    print(f"      Rest: mean={stats['middle_mean']:.1f} dBm, σ={stats['middle_std']:.2f}")
                
                if q['reasons']:
                    print(f"    Issues Detected:")
                    for reason in q['reasons']:
                        print(f"      - {reason}")
        
        print(f"{'='*70}")
        
        self.fig.canvas.draw()
    
    def next_cycle(self, event=None):
        """Move to next cycle."""
        if self.current_cycle_idx < len(self.cycle_dirs) - 1:
            self.current_cycle_idx += 1
            self.update_plot()
    
    def prev_cycle(self, event=None):
        """Move to previous cycle."""
        if self.current_cycle_idx > 0:
            self.current_cycle_idx -= 1
            self.update_plot()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'right' or event.key == 'down':
            self.next_cycle()
        elif event.key == 'left' or event.key == 'up':
            self.prev_cycle()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
    
    def show(self):
        """Display the interactive plot."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive inspector for state spectra across cycles',
        epilog='''
Navigation: Use arrow keys (← →) or buttons to move between cycles. Press Q to quit.

Reference spectrum format:
  --reference-spectrum cycle_010:1    (use spectrum 1 from cycle_010...)
  --reference-spectrum cycle_001:2    (use spectrum 2 from cycle_001...)
        ''')
    parser.add_argument('day_dir', type=Path,
                        help='Path to day directory (e.g., 20251102)')
    parser.add_argument('--state', type=str, required=True,
                        help='State number to inspect')
    parser.add_argument('--filter', type=int, default=10,
                        help='Filter number to plot (0-20, default: 10)')
    parser.add_argument('--reference-spectrum', type=str, metavar='CYCLE:IDX',
                        help='Global reference spectrum (e.g., cycle_010:1 or cycle_001_11022025_000200:2)')
    
    args = parser.parse_args()
    
    if not args.day_dir.exists():
        print(f"Error: Directory not found: {args.day_dir}")
        return 1
    
    if args.filter < 0 or args.filter > 20:
        print(f"Error: Filter number must be 0-20")
        return 1
    
    inspector = SpectrumInspector(args.day_dir, args.state, args.filter, args.reference_spectrum)
    inspector.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
