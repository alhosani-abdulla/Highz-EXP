import sqlite3
import pandas as pd
from zoneinfo import ZoneInfo
import numpy as np

def get_T_data(T_df, ts, local_tz=ZoneInfo('America/New_York'), df_colname='local_ts') -> np.ndarray:
    """Extract temperature data from dataframe given wanted time range.
    Usually thermocron data is in EST.
    
    Parameters:
        - T_df: DataFrame of thermocron T readings with a timestamp column.
        - ts: The target timestamp array (in UTC) for which to find the closest reading.
        - local_tz: The timezone of thermocron T readings.
        - df_colname: The name of timestampcolumn in thermocron readings.
        
    Return:
        - np.ndarray: temperature readings corresponding to the target timestamps."""
    if df_colname not in T_df.columns:
        raise ValueError(f"Column '{df_colname}' not found in the DataFrame.")
    
    # Prepare temperature timestamps for a vectorized nearest-neighbor match.
    temp_df = T_df[[df_colname, 'value_c']].copy()
    temp_df[df_colname] = pd.to_datetime(temp_df[df_colname], errors='coerce')

    if temp_df[df_colname].dt.tz is None:
        temp_df[df_colname] = temp_df[df_colname].dt.tz_localize(
            local_tz,
            ambiguous='NaT',
            nonexistent='NaT',
        )
    else:
        temp_df[df_colname] = temp_df[df_colname].dt.tz_convert(local_tz)

    temp_df = temp_df.dropna(subset=[df_colname, 'value_c'])
    if temp_df.empty:
        raise ValueError('No valid temperature readings found in T_df.')

    temp_df['_ts_utc'] = temp_df[df_colname].dt.tz_convert('UTC')
    temp_df = temp_df.sort_values('_ts_utc')
    temp_df['_value_k'] = temp_df['value_c'] + 273.15

    # Convert the target timestamps to UTC and preserve their original order.
    target_df = pd.DataFrame({
        '_row_id': np.arange(len(ts)),
        '_target_utc': pd.to_datetime(pd.Series(ts), utc=True, errors='coerce'),
    })
    valid_targets = target_df.dropna(subset=['_target_utc']).sort_values('_target_utc')

    matched = pd.merge_asof(
        valid_targets,
        temp_df[['_ts_utc', '_value_k']],
        left_on='_target_utc',
        right_on='_ts_utc',
        direction='nearest',
    )

    # Fill the result array in the original ts order.
    T_readings = np.full(len(target_df), np.nan, dtype=float)
    T_readings[matched['_row_id'].to_numpy()] = matched['_value_k'].to_numpy(dtype=float)

    return T_readings

class databaseReader():
    def __init__(self, db_path, mode='load'):
        """Initialize the database reader with the path to a database file and mode.
        
        Parameters:
            - `mode`: 'load' to read data, 'inspect' to print table names and structure."""
        self.db_path = db_path
        self.mode = mode
    
    def get_tables(self):
        """Return a list of table names in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if self.mode == 'inspect':
                print("Tables in the database:")
                for table in tables:
                    print(f"- {table[0]}")
            return tables
        
    def get_table_structure(self, table_name):
        """Return the structure of a specific table as a list of column info."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            if self.mode == 'inspect':
                print(f"Structure of {table_name}:")
                for col in columns:
                    print(f"Column: {col[1]}, Type: {col[2]}, Nullable: {not col[3]}")
            return columns

    def get_table_data(self, table_name, limit=7):
        """Return the data from a specific table as a pandas DataFrame, limited to a certain number of rows."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit};", conn)
            if self.mode == 'inspect':
                print(f"First 5 rows of {table_name}:")
                print(df.head(5))
            return df
    
    def get_T_readings(self, table_name='temperature_readings', deploy_id=4, sensor_id=3):
        """Convenience method to get temperature readings from the specified table."""
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT * FROM {table_name}"
            if deploy_id:
                query += f" WHERE deployment_id = {deploy_id}"
            if sensor_id:
                query += f" AND sensor_id = {sensor_id}"
            df = pd.read_sql_query(query, conn)
            if self.mode == 'inspect':
                print(f"Temperature readings from {table_name}:")
                print(df.head(5))
            return df