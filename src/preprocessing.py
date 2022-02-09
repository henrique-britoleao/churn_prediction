###############################################################################
#####  Data Preprocesssing
###############################################################################
import pandas as pd

class Preprocessor:
    def __init__(self) -> None:
        pass
    
    def preprocess(self, data:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def enforce_data_quality(self, data:pd.DataFrame) -> pd.DataFrame:
        # remove zero/negative quantities
        data = data[data['quantity'] > 0]
        # remove zero/negative sales
        data = data[data['sales_net'] > 0]
        # enforce date types
        data['date_order'] = pd.to_datetime(data['date_order'])
        data['date_invoice'] = pd.to_datetime(data['date_invoice'])
        
        data = data[
            (data['date_order'].dt.year >=2015) & 
            (data['date_invoice'].dt.year >=2015)
        ]
        
        return data
        