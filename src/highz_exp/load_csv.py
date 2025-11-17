# This file contains functions to load and process CSV files saved from Spectrum Analyzer
import csv, os
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

import csv
def parse_trace_data(data_file):
    """
    Parses a CSV file containing multiple trace blocks.
    
    Each trace block starts with 'Trace Name' and contains:
    - Metadata rows (key-value pairs) before 'Trace Data'
    - Two columns of data after 'Trace Data': frequency and spectrum
    
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
    
    # Find all rows with 'Trace Name'
    trace_starts = []
    for i, row in enumerate(rows):
        if any('Trace Name' in str(cell) for cell in row):
            trace_starts.append(i)
    
    if not trace_starts:
        print("No 'Trace Name' found in the file.")
        return []
    
    # Add end marker
    trace_starts.append(len(rows))
    
    # Parse each trace block
    traces = {}
    for idx in range(len(trace_starts) - 1):
        start = trace_starts[idx]
        end = trace_starts[idx + 1]
        block = rows[start:end]
        
        # Find 'Trace Data' row within this block
        trace_data_idx = None
        for i, row in enumerate(block):
            if any('Trace Data' in str(cell) for cell in row):
                trace_data_idx = i
                break
        
        if trace_data_idx is None:
            print(f"Warning: 'Trace Data' not found in trace block starting at row {start}")
            continue
        
        # Parse metadata (rows between 'Trace Name' (included) and 'Trace Data')
        metadata = {}
        for row in block[0:trace_data_idx]: 
            if len(row) >= 2 and row[0].strip():  # Ensure valid key-value pair
                key = row[0].strip()
                value = row[1].strip() if len(row) > 1 else ''
                metadata[key] = value
        
        tracename = metadata.get('Trace Name', 'Unknown')
        print(f"Parsing trace: {tracename}")
        metadata.pop('Trace Name')
        
        # Parse trace data (rows after 'Trace Data') using pandas
        data_rows = block[trace_data_idx + 1:]
        if data_rows:
            df = pd.DataFrame(data_rows, columns=['frequency', 'spectrum'])
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            traces[tracename] = {
                'metadata': metadata,
                'frequency': df['frequency'].to_numpy(),
                'spectrum': df['spectrum'].to_numpy()
            }
        else:
            print(f"Warning: No data found for trace {tracename}")
            traces[tracename] = {
                'metadata': metadata,
                'frequency': [],
                'spectrum': []
            }
    
    print(f"Parsed {len(traces)} trace block(s)")
    return traces