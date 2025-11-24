import copy
import warnings
import numpy as np
import pandas as pd
import parameters as parameters

# ---- Warning filters ----
warnings.filterwarnings("ignore")  # Suppress all warnings for cleaner logs

class Utils:
    """
    Utility class for handling data loading, preprocessing, and coverage statistics.
    Designed to work with RAPIDS datasets and configurable algorithms/features.
    """

    def __init__(self, folder_path="/mnt/hdd/sohini-globem-work/GLOBEM/", algo="dl_feat", category="W", dataset_number=1):
        """
        Initialize class attributes and configuration parameters.

        Args:
            folder_path (str): Base path for datasets.
            algo (str): Algorithm key to select feature subset.
            category (str): Dataset category identifier (e.g., 'W').
            dataset_number (int): Dataset index or identifier.
        """
        self.folder_path = folder_path
        self.day_name = parameters.day_name
        self.feature_categories = parameters.feature_categories
        self.algo_dict = parameters.algo_dict
        self.algo = algo
        self.category = category
        self.dataset_number = dataset_number

        # Placeholders for user-level and dataset-level data
        self.users = []
        self.rapids = pd.DataFrame()
        self.rapids_subset = pd.DataFrame()

    def load_and_preprocess(self):
        """
        Loads RAPIDS dataset, maps dates to indices, filters features,
        and prepares user and feature subsets for downstream analysis.

        Returns:
            rapids (pd.DataFrame): Full RAPIDS dataset.
            rapids_subset (pd.DataFrame): Subset with selected features.
            date_map (dict): Mapping from date to integer index.
            index_to_dates (dict): Reverse mapping from index to date.
            users (list): List of user IDs.
            user_to_id (dict): Mapping from user ID to numeric ID.
        """
        # Load dataset from CSV
        rapids = pd.read_csv(
            self.folder_path + f"data_raw/INS-{self.category}_{self.dataset_number}/FeatureData/rapids.csv",
            keep_default_na=True
        )
        print("Dataset length:", len(rapids))

        # Create date mappings
        dates = sorted(set(rapids.date))
        date_map = {date: i for i, date in enumerate(dates)}
        index_to_dates = dict(enumerate(dates))

        rapids["date_map"] = rapids.date.replace(date_map)

        # Deep copy for processing
        rapids_ = copy.deepcopy(rapids)

        # Define main feature categories
        self.feature_categories = ["f_call", "f_blue", "f_wifi", "f_slp", "f_steps", "f_loc", "f_screen"]

        # Remove unnecessary columns (e.g., unnamed or distance-related)
        if "Unnamed: 0" in rapids_.columns:
            rapids_ = rapids_.drop(["Unnamed: 0"], axis=1)
        rapids_ = rapids_.drop(rapids_.filter(regex="_dis:"), axis=1)

        # Extract user list
        users = list(rapids_.pid.unique())

        # Filter based on selected algorithm’s feature set
        self.algo_dict[self.algo] = set(self.algo_dict[self.algo])
        features = set(self.algo_dict[self.algo])
        print("Number of features:", len(features))

        rapids_subset = rapids_[['pid', 'date_map'] + list(features)]

        # Create user ↔ ID mapping
        user_to_id = {k: v for v, k in enumerate(users)}

        # Store results as class attributes
        self.users = users
        self.rapids = rapids
        self.rapids_subset = rapids_subset

        return rapids, rapids_subset, date_map, index_to_dates, users, user_to_id

    def get_not_null_percentages(self, df=[]):
        """
        Calculates the percentage of non-null (observed) values per feature category for each user.

        Args:
            df (pd.DataFrame, optional): Data subset to evaluate. Defaults to self.rapids_subset.

        Returns:
            pd.DataFrame: DataFrame with users and their non-null percentages by feature category.
        """
        users = self.users
        if not df:
            df = self.rapids_subset.copy()

        df_full = pd.DataFrame(users, columns=['pid'])

        # Iterate through feature categories
        for c in ["_"] + self.feature_categories:
            data = df.filter(regex=c + "|^pid$")

            # Skip categories with ≤ 1 column
            if len(data.columns) <= 1:
                print(c, "has ≤ 1 column, skipping.")
                continue

            # Compute null and non-null counts per user
            null_count = data.drop('pid', axis=1).isnull().groupby(data.pid, sort=False).sum()
            null_count["total_null_sum_" + c] = null_count.sum(axis=1)
            null_count.reset_index(inplace=True)

            notnull_count = data.drop('pid', axis=1).notnull().groupby(data.pid, sort=False).sum()
            notnull_count["total_notnull_sum_" + c] = notnull_count.sum(axis=1)
            notnull_count.reset_index(inplace=True)

            # Combine and compute percentages
            notnull_count["total_" + c] = (
                null_count["total_null_sum_" + c] + notnull_count["total_notnull_sum_" + c]
            )
            notnull_count["percentage_" + c] = (
                notnull_count["total_notnull_sum_" + c] / notnull_count["total_" + c]
            ) * 100

            # Merge with master DataFrame
            percentages_df = notnull_count[["percentage_" + c]]
            df_full = pd.concat([df_full, percentages_df], axis=1)

        return df_full

    def calc_coverage(self):
        """
        Computes and prints the average feature coverage (non-null ratio)
        across all users and features.
        """
        df = self.rapids_subset.copy()
        coverages = []

        # Calculate coverage for each feature
        for col in self.algo_dict[self.algo]:
            coverage = df[col].notnull().sum()
            coverages.append((col, coverage))

        # Sort and calculate average coverage ratio
        sorted_coverages = sorted(coverages, key=lambda x: x[1], reverse=True)
        total_coverage = [a[1] for a in sorted_coverages]
        print(sum(total_coverage) / (len(self.algo_dict[self.algo]) * len(df)))
