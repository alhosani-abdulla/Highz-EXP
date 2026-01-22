# This file contains functions to load and process CSV files saved from Spectrum Analyzer SAX3000.
import csv, os
import numpy as np
import pandas as pd

pjoin = os.path.join
pbase = os.path.basename

def split_csv_by_trace_name(input_file, header_file=None, data_file=None):
    """
    Splits a CSV file into two files based on the first occurrence of 'Trace Name'.
    
    Args:
        input_file: Path to the input CSV file
        header_file: Path for the output header CSV file (rows before 'Trace Name')
        data_file: Path for the output data CSV file (rows from 'Trace Name' onward)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
      
    parent_dir = os.path.dirname(input_file)
    input_name = pbase(input_file)
    if header_file is None:
        header_file = pjoin(parent_dir, 'Meta_' + input_name)
    if data_file is None:
        data_file = pjoin(parent_dir, 'Data_' + input_name)

    # Find the first row containing "Trace Name"
    split_index = None
    for i, row in enumerate(rows):
        if any('Trace Name' in str(cell) for cell in row):
            split_index = i
            break
    
    if split_index is None:
        print("'Trace Name' not found in the CSV file.")
        return
    
    # Split the rows
    header_rows = rows[:split_index]
    data_rows = rows[split_index:]
    
    # Write header file
    with open(header_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(header_rows)
    
    # Write data file
    with open(data_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)
    
    print(f"Split complete:")
    print(f"  Header file: {header_file} ({len(header_rows)} rows)")
    print(f"  Data file: {data_file} ({len(data_rows)} rows)")

def parse_trace_data(data_file):
    """
    Parses a CSV file containing multiple traces in column-oriented format.
    
    Structure:
    - First row contains headers: 'Trace Name', 'Trace A', 'Trace Name', 'Trace B', etc.
    - Metadata rows follow (Trace Type, Trace Detector, etc.)
    - 'Trace Data' row marks the start of frequency/spectrum columns
    - Data rows contain frequency and spectrum values for each trace in pairs
    
    Args:
        data_file: Path to the data CSV file
        
    Returns:
      Dictionary where keys are trace names and values are dictionaries containing:
        - 'metadata': dict of key-value pairs
        - 'frequency': np.array of frequency values
        - 'spectrum': np.array of spectrum values
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Parse header row to extract trace names
    header_row = rows[0]
    trace_names = []
    trace_col_pairs = []  # Store (freq_col, spec_col) pairs
    
    # Extract trace names and their column positions
    # Pattern: 'Trace Name', 'Trace A', 'Trace Name', 'Trace B', etc.
    for i in range(1, len(header_row), 2):
        if i + 1 < len(header_row):
            trace_name = header_row[i].strip()
            if trace_name:
                trace_names.append(trace_name)
                trace_col_pairs.append((i - 1, i))  # (frequency col, spectrum col)
    
    print(f"Found {len(trace_names)} traces: {trace_names}")
    print(f"Column pairs: {trace_col_pairs}")
    
    if not trace_names:
        print("No traces found in the file.")
        return {}
    
    # Find the 'Trace Data' row
    trace_data_row_idx = None
    for i, row in enumerate(rows):
        if any('Trace Data' in str(cell) for cell in row):
            trace_data_row_idx = i
            break
    
    if trace_data_row_idx is None:
        print("Warning: 'Trace Data' marker not found in file")
        trace_data_row_idx = 3  # Default assumption
    
    print(f"'Trace Data' found at row {trace_data_row_idx}")
    
    # Parse metadata rows (between header and 'Trace Data')
    metadata_dict = {}
    for row_idx in range(1, trace_data_row_idx):
        row = rows[row_idx]
        if len(row) >= 2:
            key = row[0].strip()
            if key and key not in ['Trace Name']:
                metadata_dict[key] = row[1].strip() if len(row) > 1 else ''
    
    # Parse data rows (after 'Trace Data')
    traces = {}
    
    for trace_name, (freq_col, spec_col) in zip(trace_names, trace_col_pairs):
        frequencies = []
        spectra = []
        
        # Extract data from the specified column pair (freq, spectrum)
        for row_idx in range(trace_data_row_idx + 1, len(rows)):
            row = rows[row_idx]
            
            # Get frequency and spectrum values
            if freq_col < len(row) and spec_col < len(row):
                freq_str = row[freq_col].strip()
                spec_str = row[spec_col].strip()
                
                if freq_str and spec_str:
                    try:
                        freq = float(freq_str)
                        spec = float(spec_str)
                        frequencies.append(freq)
                        spectra.append(spec)
                    except ValueError:
                        continue
        
        print(f"Trace '{trace_name}': parsed {len(frequencies)} data points")
        
        if frequencies and spectra:
            traces[trace_name] = {
                'metadata': metadata_dict.copy(),
                'frequency': np.array(frequencies),
                'spectrum': np.array(spectra)
            }
    
    print(f"Successfully parsed {len(traces)} traces with data")
    return traces
