import os
import pandas as pd
import os
import pathlib
import tempfile
from zipfile import ZipFile
import requests
import shutil

from huggingface_hub import hf_hub_download
import pickle


INPUT_DATA_DIR = "ml-ddd/data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(mkdir=True, verbose=False):
    url = MOVIELENS_URL
    if verbose is True:
        print(f"Downloading from {url}")
    output_dir = pathlib.Path(INPUT_DATA_DIR).resolve()
    if not output_dir.exists():
        if mkdir:
            output_dir.mkdir(exist_ok=True)
        else:
            raise Exception(f"{output_dir} does not exist. Pass `mkdir=True`")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        with tempfile.NamedTemporaryFile(mode='rb+') as temp_f:
            downloaded = 0
            dl_iteration = 0
            chunk_size = 8192
            total_chunks = total_size_in_bytes / chunk_size if total_size_in_bytes else 100
            for chunk in r.iter_content(chunk_size=chunk_size):
                if verbose is True:
                    downloaded += chunk_size
                    dl_iteration += 1
                    percent = (100 * dl_iteration * 1.0/total_chunks)
                    if dl_iteration % 10 == 0 and percent < 100:
                        print(f'Completed {percent:2f}%')
                    elif percent >= 99.9:
                        print(f'Download completed. Now unzipping...')
                temp_f.write(chunk)
            with ZipFile(temp_f, 'r') as zipf:
                zipf.extractall(output_dir)
                if verbose is True:
                    print(f"\n\nUnzipped.\n\nFiles downloaded and unziped to:\n\n{INPUT_DATA_DIR.resolve()}")



def download_ml_ratings():
    print("DOWNLOADING MOVIELENS 25M...")

    download_movielens()

    ratings_path = os.path.join(os.path.join(INPUT_DATA_DIR, 'ml-25m'), 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)

    print("COMPLETED DOWNLOAD")
    return ratings_df


def download_ddd_warnings():
    print("DOWNLOADING DOES THE DOG DIE WARNING DICTIONARY...")
    repo_id = "sdeangroup/NavigatingSensitivity"
    ddd_dict = "ddd_dict.pkl"

    ddd_path = open(hf_hub_download(repo_id=repo_id, filename=ddd_dict, repo_type="dataset"), "rb")
    ddd_destination = open(f"{INPUT_DATA_DIR}/ddd_dict.pkl", "wb")
    shutil.copyfileobj(ddd_path, ddd_destination)

    with open(f"{INPUT_DATA_DIR}/ddd_dict.pkl", 'rb') as handle:
        ddd_dict = pickle.load(handle)

    print("COMPLETED DOWNLOAD")
    return ddd_dict


def get_warning_votes(votes):
    """
    Returns 1 hot list: clear yes (yesSum > 75% total), clear no (noSum > 75% total), unclear (yesSum < 75% and noSum < 75%), no votes (total = 0)
    """
    majority = 0.75 * (votes['yesSum'] + votes['noSum'])
    if majority == 0:
        # no votes
        return 0, 0, 0, 1
    elif votes['yesSum'] > majority:
        # majority yes votes
        return 1, 0, 0, 0
    elif votes['noSum'] > majority:
        # majority no votes
        return 0, 1, 0, 0
    else:
        # no clear consensus
        return 0, 0, 1, 0
    

def get_sensitivity_table(ddd_dict):
    data = {}
    warnings = set()
    for warning, work_votes in ddd_dict.items():
        if warning not in warnings: warnings.add(warning)
        for work_id, votes in work_votes.items():
            if work_id not in data:
                data[work_id] = {}
            data[work_id][f"Clear Yes: {warning}"], data[work_id][f"Clear No: {warning}"], data[work_id][f"Unclear: {warning}"], data[work_id][f"No Votes: {warning}"] = get_warning_votes(votes)

    # Creating DataFrame
    sensitivity_table = pd.DataFrame(data).T.fillna(0).astype(int)
    sensitivity_table.reset_index(inplace=True)
    sensitivity_table.rename(columns={'index': 'work_id'}, inplace=True)

    return sensitivity_table


def filter_tables(sensitivity_table, interaction_table):
    """
    Filter the dataframes to only contain users with at least 3 interactions and works with at least 3 interactions.
    """
    interaction_table = interaction_table.rename(columns={"userId": "user_id", "movieId": "work_id"})
    print(f"Initial number of users before filtering: {len(interaction_table['user_id'].unique())}")
    print(f"Initial number of works before filtering: {len(sensitivity_table)}")

    for i in range(3):
        # Remove users from interaction_table with less than three interactions
        user_counts = interaction_table['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= 3].index
        interaction_table = interaction_table[interaction_table['user_id'].isin(users_to_keep)]
        existing_users = set(interaction_table["user_id"].unique())
        print(f"Number of users with at least 3 interactions (pass {i}): {len(existing_users)}")

        # Remove works that appear less than three times in the interaction_table
        interactions = interaction_table['work_id'].value_counts()
        works_to_keep = interactions[interactions >= 3].index
        sensitivity_table = sensitivity_table[sensitivity_table["work_id"].isin(works_to_keep)]
        print(f"Number of works with at least 3 likes (pass {i}): {len(sensitivity_table)}")
            
        # Update interaction_table to include only works that are present after filtering users and works
        works_to_keep = sensitivity_table['work_id'].unique()
        interaction_table = interaction_table[interaction_table['work_id'].isin(works_to_keep)]

    print(f"Final number of users: {len(interaction_table['user_id'].unique())}")
    print(f"Final number of works: {len(sensitivity_table)}")
    print(f"Final number of interactions: {len(interaction_table)}")

    return sensitivity_table, interaction_table


def add_summary_stats(sensitivity_table, interaction_table):
    # Calculate the number of ratings (n_ratings) and average rating (av_rating) for each work
    work_ratings_summary = interaction_table.groupby('work_id').agg(n_ratings=('rating', 'count'), av_rating=('rating', 'mean')).reset_index()

    sensitivity_table = pd.merge(sensitivity_table, work_ratings_summary, on='work_id', how='left')
    user_dict = interaction_table.groupby('work_id').apply(lambda x: {user: rating for user, rating in zip(x['user_id'], x['rating'])}).to_dict()

    def get_ratings(work_id):
        return user_dict.get(work_id, {})

    # Add dictionary of each user and their rating for each each
    sensitivity_table['user_ratings'] = sensitivity_table['work_id'].map(get_ratings)

    return sensitivity_table


if __name__ == "__main__":
    # Create data folder
    if not os.path.exists(INPUT_DATA_DIR):
        os.makedirs(INPUT_DATA_DIR)
    ratings_df = download_ml_ratings()
    ddd_dict = download_ddd_warnings()
    sensitivity_table = get_sensitivity_table(ddd_dict)
    sensitivity_table, interaction_table = filter_tables(sensitivity_table, ratings_df)
    sensitivity_table = add_summary_stats(sensitivity_table, interaction_table)

    sensitivity_table.to_csv(os.path.join(INPUT_DATA_DIR, "ml-ddd_sensitivity_table.csv"), index=False)
    interaction_table.to_csv(os.path.join(INPUT_DATA_DIR, "ml-ddd_interaction_table.csv"), index=False)