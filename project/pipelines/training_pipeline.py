#from zenml.config import DockerSettings
#from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

#docker_settings = DockerSettings(required_integrations=[MLFLOW])


def data_path() -> str:
    return r"C:\Users\hp\OneDrive\Desktop\projects\resume_projects\ML pipeline(zenml)\data\merged_data.csv"

@pipeline(enable_cache=True)
def train_pipeline():
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        r2_score: float
        rmse: float
    """
    data_path_value = data_path()
    df = ingest_df(data_path_value)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, x_test, y_test)