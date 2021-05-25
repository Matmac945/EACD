# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config(path="./.azureml", _file_name="config.json")

    model = Model.register(
        model_name="colombian_tweet_clf",
        tags={"area": "udea_cloud_final"},
        model_path="outputs/colombian_tweet_clf.joblib",
        workspace=ws,
    )
    print(model.name, model.id, model.version, sep="\t")

