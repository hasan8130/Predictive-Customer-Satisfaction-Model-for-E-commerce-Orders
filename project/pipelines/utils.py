import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

## preparing our testing data
def get_data_for_test():
    try:
        df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\projects\resume_projects\ML pipeline(zenml)\data\merged_data.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e