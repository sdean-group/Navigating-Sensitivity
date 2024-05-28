from huggingface_hub import hf_hub_download
import pandas as pd
import shutil
import os

INPUT_DATA_DIR = "ao3/data"

def download_ao3_data():
    print("DOWNLOADING AO3 DATA...")
    repo_id = "sdeangroup/NavigatingSensitivity"
    sensitivity_table = "ao3_sensitivity_table.csv"
    interaction_table = "ao3_interaction_table.csv"

    sensitivity_path = open(hf_hub_download(repo_id=repo_id, filename=sensitivity_table, repo_type="dataset"), "rb")
    sensitivity_destination = open(f"{INPUT_DATA_DIR}/{sensitivity_table}", "wb")
    shutil.copyfileobj(sensitivity_path, sensitivity_destination)

    interaction_path = open(hf_hub_download(repo_id=repo_id, filename=interaction_table, repo_type="dataset"), "rb")
    interaction_destination = open(f"{INPUT_DATA_DIR}/{interaction_table}", "wb")
    shutil.copyfileobj(interaction_path, interaction_destination)

    print("COMPLETED DOWNLOAD")


if __name__ == "__main__":
    # Create data folder
    if not os.path.exists(INPUT_DATA_DIR):
        os.makedirs(INPUT_DATA_DIR)
    download_ao3_data()