import dask.dataframe as dd
import pandas as pd

def load_and_preprocess(file_path):
    # Load the dataset
    data = dd.read_csv(file_path)
    
    # Select necessary columns
    data = data[["Country/Region", "Confirmed", "Deaths", "Recovered", "Active"]]
    
    # Calculate Recovery Rate and Death Rate
    data["Recovery Rate (%)"] = (data["Recovered"] / data["Confirmed"]) * 100
    
    # Convert to Pandas for further processing
    data = data.compute()
    
    # Clean the data
    data = data.replace([float("inf"), -float("inf")], None).dropna()
    
    return data
