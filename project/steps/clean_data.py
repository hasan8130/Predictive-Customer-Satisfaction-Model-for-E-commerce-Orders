import logging
import pandas as pd
from zenml import step 
from typing import Tuple
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessStrategy
from typing_extensions import Annotated


@step
def clean_df(df: pd.DataFrame) -> Tuple[ Annotated[pd.DataFrame, "x_train"],Annotated[pd.DataFrame, "x_test"],Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    
    Return:
    x_train, x_test ,y_train ,y_test
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e