"""
Filter Response Plotting and Analysis Tools

Utilities for loading and converting filterbank calibration data,
converting ADC counts to voltages and power, and creating filter response plots.
"""

import numpy as np
import re
from astropy.io import fits


def _to_str(v):
    """Convert bytes or other types to string"""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="ignore")
    return str(v)


def _parse_freq_mhz(s):
    """Parse frequency string and return value in MHz"""
    if s is None:
        return np.nan
    s = _to_str(s).strip()
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    low = s.lower()
    if "ghz" in low:
        return val * 1e3
    elif "mhz" in low:
        return val
    elif "khz" in low:
        return val / 1e3
    elif "hz" in low:
        return val / 1e6
    return val  # assume MHz


def load_filterbank_table(path, hdu_index=1,
                          cols=("ADHAT_1", "ADHAT_2", "ADHAT_3"),
                          freq_col="FREQUENCY"):
    """
    Load filterbank calibration table from FITS file.
    
    Parameters
    ----------
    path : str
        Path to FITS file
    hdu_index : int
        HDU index to load (default: 1 for binary table)
    cols : tuple
        Column names containing ADC data (default: ADHAT_1, ADHAT_2, ADHAT_3)
    freq_col : str
        Column name for frequency data (default: FREQUENCY)
    
    Returns
    -------
    lo_mhz : ndarray
        LO frequencies in MHz (n_steps,)
    data21 : ndarray
        21 filter responses (n_steps, 21)
    """
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[hdu_index]
        assert hdu.header.get("XTENSION", "").upper() in {"BINTABLE", "TABLE"}
        tbl = hdu.data
        blocks = []
        
        for c in cols:
            cname = next((nm for nm in hdu.columns.names if nm.upper() == c.upper()), None)
            if cname is None:
                raise KeyError(f"Column '{c}' not found. Available: {hdu.columns.names}")
            arr = np.asarray([np.ravel(x) for x in np.asarray(tbl[cname])])  # (n_steps, width)
            blocks.append(arr)
        
        data21 = np.concatenate(blocks, axis=1).astype(float)  # (n_steps, 21)

        cfreq = next((nm for nm in hdu.columns.names if nm.upper() == freq_col.upper()), None)
        lo_mhz = (np.array([_parse_freq_mhz(v) for v in tbl[cfreq]], dtype=float)
                  if cfreq is not None else np.arange(len(tbl), dtype=float))
        
        return lo_mhz, data21


def adc_counts_to_voltage(counts, ref=3.27, mode="c_like",
                          denom_pos=2147483647.8, denom_neg=2147483648.0):
    """
    Convert ADS1263 ADC counts to Volts.
    
    Parameters
    ----------
    counts : array-like
        Raw ADC codes (often stored as floats in files)
    ref : float
        Reference voltage (Volts). Default 3.27 V, use 5.0 for calibration data
    mode : {"c_like", "signed_bipolar"}
        - "c_like": replicate C-style mapping, resulting in ~0..ref V
        - "signed_bipolar": interpret as true signed int32 (±ref full-scale)
    denom_pos, denom_neg : float
        Denominators used in C-like mapping (kept tunable)
    
    Returns
    -------
    V : ndarray
        Volt values (same shape as counts)
    """
    c = np.asarray(counts)

    if mode == "c_like":
        # Work with float arrays; derive sign from MSB as if counts were uint32
        cu = c.astype(np.uint64)
        neg = ((cu >> 31) & 0x1) == 1  # MSB set

        V = np.empty_like(c, dtype=float)
        # Negative branch (MSB=1)
        V[neg] = ref * 2.0 - (cu[neg].astype(float) / denom_neg) * ref
        # Positive branch (MSB=0)
        V[~neg] = (cu[~neg].astype(float) / denom_pos) * ref
        return V

    elif mode == "signed_bipolar":
        # Interpret as true signed int32 counts (two's complement)
        cs = c.astype(np.int64)  # safe up-cast
        cs = ((cs + (1 << 31)) % (1 << 32)) - (1 << 31)
        # Map ±(2^31-1) -> ±ref
        V = (cs / float((1 << 31) - 1)) * ref
        return V

    else:
        raise ValueError("mode must be 'c_like' or 'signed_bipolar'.")


def voltage_to_dbm(V, R=50.0, assume="rms"):
    """
    Convert Volts to power in dBm for a resistive load.
    
    Parameters
    ----------
    V : array-like
        Volt values
    R : float
        Load resistance in ohms (default: 50 Ω)
    assume : {"rms", "peak"}
        If "peak", convert peak to Vrms first
    
    Returns
    -------
    dBm : ndarray
        Power in dBm (non-positive values map to -inf)
    """
    V = np.asarray(V, dtype=float)
    Vrms = V if assume.lower() == "rms" else (V / np.sqrt(2.0))
    P_w = (Vrms ** 2) / R
    with np.errstate(divide="ignore", invalid="ignore"):
        dbm = 10.0 * np.log10(P_w / 1e-3)
    dbm[~np.isfinite(dbm)] = -np.inf
    return dbm


def get_filter_centers(num_filters=21, start_mhz=904.0, step_mhz=2.6):
    """
    Get center frequencies for filterbank.
    
    Parameters
    ----------
    num_filters : int
        Number of filters (default: 21)
    start_mhz : float
        Starting center frequency in MHz (default: 904.0)
    step_mhz : float
        Frequency step between filters in MHz (default: 2.6)
    
    Returns
    -------
    centers : ndarray
        Center frequencies in MHz (num_filters,)
    """
    return start_mhz + step_mhz * np.arange(num_filters)
