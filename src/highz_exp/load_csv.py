# This file contains functions to load and process CSV files saved from Spectrum Analyzer
import csv, os

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