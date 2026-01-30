import pandas as pd
from pathlib import Path


def load_csvs_from_folder(data_folder: str) -> dict:
    """
    Load all CSV files from a specified folder into a dictionary.
    
    Parameters
    ----------
    data_folder : str
        Path to the folder containing CSV files
    
    Returns
    -------
    dict
        Dictionary with CSV filenames (without extension) as keys 
        and DataFrames as values
    """
    data_dict = {}
    data_path = Path(data_folder)
    
    for csv_file in data_path.glob('*.csv'):
        file_name = csv_file.stem
        df_temp = pd.read_csv(csv_file,index_col=0)
        df_temp.index = pd.to_datetime(df_temp.index).normalize()
        data_dict[file_name] = df_temp
    
    return data_dict
