import pandas as pd

def describe_columns(data):
    total = data.count()
    xx = pd.DataFrame(total)
    xx.columns = ['Total']
    
    data_types = []
    uniques = []
    most_frequent = [] 
    least_frequent = []
    missing = []
    
    for col in data.columns:
        uniques.append(data[col].nunique())
        data_types.append(data[col].dtype)
        
        mode_series = data[col].mode()
        most_frequent.append(mode_series[0] if not mode_series.empty else None)
        
        value_counts = data[col].value_counts()
        least_frequent.append(value_counts.idxmin() if not value_counts.empty else None)
        
        missing.append(data[col].isnull().sum())
    
    xx['Uniques'] = uniques
    xx['Missing'] = missing
    xx['DataType'] = data_types
    xx['MostFrequent'] = most_frequent
    xx['LeastFrequent'] = least_frequent
    
    return xx