# Navigating-Sensitivity

This repository demonstrates how to download and process two novel datasets for recommendation system training and evaluation. Each dataset contains two files: 1) A **sensitivity_table**, enumerating item identifiers, associated content warnings, and various metadata, and 2) an **interaction_table**, listing the observed ratings or likes between users and works.

Our data is hosted on Hugging Face: https://huggingface.co/datasets/sdeangroup/NavigatingSensitivity

We provide python scripts for downloading and processing it and jupyter notebooks for analyzing it.

## ML/DDD
This dataset augments rating data from the [MovieLens 25m dataset](https://grouplens.org/datasets/movielens/) [1] with content warnings collected from [Does the Dog Die?](https://www.doesthedogdie.com/), a community-driven platform for determining trigger warnings for media through a voting system. We provide a pickled dictionary, `ddd.pkl`, throuh Hugging Face with movies and their associated warnings. There are 137 distinct warnings included in the DDD dictionary. To generate `ml-ddd_sensitivity_table.csv` and `ml-ddd_interaction_table.csv`, run the following:
```
python3 ml-ddd/ml-ddd_data_generation.py
```
This file will download the MovieLens 25m dataset and DDD dictionary a into the data folder and merge the two appropriately into data tables. The tables will be filtered to only contain movies that appear in the DDD dictionary and have at least three ratings, and users who have rated at least three movies.

Each row in `ml-ddd_sensitivity_table.csv` represents a movie which appears in both ML-25M and the DDD API. It has the following columns:
- `work_id `: unique numerical identifier for the work. Corresponds to the movie's ID in the MovieLens dataset.
- `n_ratings`: number of users who have rated the movie.
- `av_rating`: average rating the work received.
- `user_ratings`: dictionary of every user who rated the movie and their corresponding rating.
- `Clear Yes: <warning>`: 1 if at least 75% of votes indicated the warning applies to the movie, 0 otherwise.
- `Clear No: <warning>`: 1 if at least 75% of votes indicated the warning does not apply to the movie, 0 otherwise.
- `No Votes: <warning>`: 1 if the work received no votes about the warning, 0 otherwise.
- `Unclear: <warning>`: 1 if neither the "yes" or "no" votes made up 75% of the total number of votes, 0 otherwise.
- `all_warnings`: list of every warning marked "Clear Yes" for the work.

There is a Clear Yes, Clear No, No Votes, and Unclear column for every warning. Only one of these columns will equal 1 for a given warning.

Each row in `ml-ddd_interaction_table.csv` represents an interaction between a user and movie. We consider these interactions as explicit since they carry notions of both positive and negative preference.
It has the following columns:
- `user_id`: unique numerical identifier for the user. Corresponds to the user's ID in the MovieLens dataset.
- `work_id`: unique numerical identifier for the work. Corresponds to the movie's ID in the MovieLens dataset.
- `rating`: the user's rating for the given work on a 0.5 to 5 point scale.

The notebook `ml-ddd/ml-ddd_summary_stats.ipynb` demonstrates how to process the data and generate summary statistics and plots. The notebook also contains more descriptive details about the qualities of the dataset.
The notebook `ml-ddd/ml-ddd_recommender_analysis.ipynb` trains and evaluates three recommendation algorithms on the dataset: TopPop, Random, and SVD. It examines the amplification of warnings in recommendations.


## AO3
The AO3 dataset contains user-work interactions from [Archive of Our Own](https://archiveofourown.org/) (AO3) with content warnings from the [Webis Trigger Warning Corpus 2023](https://zenodo.org/records/7976807). `ao3_interaction_table.csv` contains publicly available user-work interactions collected from AO3 between March, 2024, and May, 2024. Usernames from the public AO3 user accounts are pseudonymized as numerical IDs. The works collected represent a subset of those presented in the Corpus. We collected updated kudos, hits, and user interactions for a third of the works in the corpus. 11% of the attempted work downloads failed due to being privatized or deleted. The works were collected sequentially at indices 0, 100,000, 200,000, 300,000, 500,000 and 700,000 of the dehydrated corpus and were confirmed as a representative subset of the dataset. `ao3_sensitivity_table.csv` contains the updated kudos, hits, and user interactions along with the trigger warning categorizations from the corpus. We focus on the 36 distinct fine closed warning categories defined in [2]. Only works with at least 3 interactions and users with at least 3 interactions are included.

`ao3_sensitivity_table.csv` and `ao3_interaction_table.csv` are hosted on our Huggng Face. To download them into the ao3 data folder run the following:
```
python3 ao3/ao3_data_generation.py
```

Each row in `ao3_sensitivity_table.csv` represents an AO3 fanfiction. It has the following columns:
- `work_id`: unique numerical identifier for the work. Corresponds to the AO3 identifier. You can access the fanfiction's page with f"https://archiveofourown.org/works/{work_id}?show_comments=true&view_adult=true&view_full_work=true".
- `n_hits`: number of "hits" a work receives. These represent the number of times a user has read the work.
- `n_kudos`: number of "kudos" a work receives. These represent likes given by users, including those on guest accounts.
- `n_users`: number of users with a public account who gave kudos to the work. We refer to this metric simply as "interactions".
- `users`: list of the users with a public account who gave kudos to the work. Usernames are pseudonomyzed as numerical identifiers.
- `<warning>`: 1 if the warning applies to the work, 0 otherwise.
- `warnings_fine_open`: all fine open warnings which apply to the work.

There is a `<warning>` column for every fine open warning.

Each row in `ao3_interaction_table.csv` represents an interaction between a public user and a work. These interactions are implicit: we only know what works a user has liked (given kudos to) and there is no notion of dislike as we cannot differentiate it from works a user has never come across. The table has the following columns:
- `user_id`: unique numerical identifier representing a public AO3 user account. These ids are not traceable to the original user accounts.
- `work_id`: unique numerical identifier for the work. Corresponds to the AO3 identifier.

The notebook `ao3/ao3_summary_stats.ipynb` demonstrates how to process the data and generate summary statistics and plots. The notebook also contains more descriptive details about the qualities of the dataset.
The notebook `ao3/ao3_recommender_analysis.ipynb` trains and evaluates three recommendation algorithms on the dataset: TopPop, Random, and ALS. It examines the amplification of warnings in recommendations.


## Installs
In order to run the data generation scripts or jupyter notebooks you need certain Python libraries installed. We suggest creating a conda environment for this project. All the packages can be installed with pip or conda.


## References
[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Trans. Interact. Intell. Syst. 5, 4, Article 19 (dec 2015), 19 pages.
https://doi.org/10.1145/2827872. 

[2] Matti Wiegmann, Magdalena Wolska, Christopher Schröder, Ole Borchardt,
Benno Stein, and Martin Potthast. 2023. Trigger warning assignment as a multi-
label document classification problem. In Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers). 12113–
12134.