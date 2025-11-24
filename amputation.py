import math
import random
import copy
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm

import parameters as parameters
from utils import Utils

import matplotlib.pyplot as plt

# Suppress irrelevant warnings
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

class Amputation():
    """
    The Amputation class handles different strategies for introducing missing values 
    (i.e., 'amputation') into user-level datasets. This is useful for 
    benchmarking imputation algorithms by simulating various missingness patterns.
    """

    def __init__(self, rapids_df, rapids_subset, users=[], algo="dl_feat"):
        """
        Initialize the Amputation class.

        Parameters
        ----------
        rapids_df : pd.DataFrame
            The complete dataset before amputation.
        rapids_subset : pd.DataFrame
            The subset of data used for amputation and evaluation.
        users : list, optional
            List of user IDs to process (default: empty list).
        algo : str, optional
            Algorithm key from parameters.algo_dict determining 
            which columns/features to use (default: "dl_feat").
        """

        self.rapids_df = rapids_df  # Full dataset
        self.rapids_subset = rapids_subset  # Subset of dataset for experiments
        self.users = users  # List of users to process

        # Load algorithm-specific configuration
        self.algo_dict = parameters.algo_dict
        self.algo = algo
        self.columns = list(self.algo_dict[self.algo])  # Select relevant feature columns

        # self.folder_path = folder_path  # (optional) For saving experiment results


    def ampute(self, amputation_type: str = "", percentage_to_ampute: float = 0.1, lower_percentile: float = 5, upper_percentile: float = 95):
        """
        Dispatcher method to perform data amputation (introducing missing values)
        based on the selected strategy.

        Parameters
        ----------
        amputation_type : str, optional
            Type of amputation to perform:
            - "" (empty string): Random missingness (MCAR) via percentage.
            - "mnar_i" : Missing Not At Random (MNAR) - middle percentile (values between bounds).
            - "mnar_ii": Missing Not At Random (MNAR) - extreme percentile (values outside bounds).
        percentage_to_ampute : float, optional
            Fraction of non-null values to ampute (used for MCAR mode).
        lower_percentile : float, optional
            Lower percentile bound for MNAR modes.
        upper_percentile : float, optional
            Upper percentile bound for MNAR modes.

        Returns
        -------
        tuple
            (amputed_df, amputed_data_indices, amputed_count, amputed_avg)
            where:
            - amputed_df : pd.DataFrame
                DataFrame with missing values introduced.
            - amputed_data_indices : dict
                Indices of values amputed per user and column.
            - amputed_count : dict
                Number of amputed values per user.
            - amputed_avg : dict
                Average amputation ratio per user.
        """

        if amputation_type == "":
            # Missing Completely At Random (MCAR)
            self.amputed_df, self.amputed_data_indices, self.amputed_count, self.amputed_avg = (
                self.ampute_by_percentage(percentage_to_ampute)
            )

        elif amputation_type == "mnar_i":
            # Missing Not At Random (MNAR) - middle percentile range
            self.amputed_df, self.amputed_data_indices, self.amputed_count, self.amputed_avg = (
                self.ampute_by_percentile_mnar_i(lower_percentile, upper_percentile)
            )

        elif amputation_type == "mnar_ii":
            # Missing Not At Random (MNAR) - extreme percentile range
            self.amputed_df, self.amputed_data_indices, self.amputed_count, self.amputed_avg = (
                self.ampute_by_percentile_mnar_ii(lower_percentile, upper_percentile)
            )

        else:
            print(f"ERROR: Unknown amputation type '{amputation_type}'. "
                  f"Valid options are '', 'mnar_i', or 'mnar_ii'.")
            return None

        # Return the results
        return (self.amputed_df, self.amputed_data_indices, self.amputed_count, self.amputed_avg)

    
    def ampute_by_percentage(self, percentage_to_ampute: float):
        """
        Randomly ampute (set to NaN) a fixed percentage of non-null values per user and column.

        This method introduces missingness by randomly removing a user-specific proportion of data,
        creating a Missing Completely At Random (MCAR) pattern.

        For each user and each numeric column:
            - Identifies all non-null entries.
            - Randomly selects a given percentage of those entries.
            - Replaces them with NaN.

        Parameters
        ----------
        percentage_to_ampute : float
            Fraction of non-null data to be amputed (e.g., 0.1 = 10%).

        Returns
        -------
        df_to_ampute : pd.DataFrame
            The DataFrame with selected values replaced by NaN.
        amputed_data_indices : dict
            Nested dictionary storing indices of amputed (NaN) values for each user and column.
        amputed_count : dict
            Dictionary mapping user → total number of amputed values.
        amputed_avg : dict
            Dictionary mapping user → average proportion of amputed values across columns.
        """

        # Columns that should not be modified
        excluded_columns = ['pid', 'date_map']

        # Copy base data
        df = self.rapids_subset.copy()
        users = self.users
        df_to_ampute = df.copy()

        # Tracking structures
        amputed_data_indices = {}
        amputed_count = {}
        amputed_avg = {}

        print(f"Amputing {percentage_to_ampute*100:.1f}% of non-null data randomly per user-column...")

        # Iterate through each user
        for user in tqdm.tqdm(users):
            user_df = df.loc[df['pid'] == user]
            amputed_data_indices[user] = {}
            amputed_count[user] = 0
            amputed_avg[user] = 0

            # Iterate through columns for this user
            for col in user_df.columns:
                amputed_data_indices[user][col] = []

                # Skip identifier columns
                if col in excluded_columns:
                    continue

                # Skip if column is entirely NaN
                col_values = user_df[col].dropna()
                if col_values.empty:
                    continue

                # Determine number of entries to ampute
                n_to_ampute = math.floor(percentage_to_ampute * user_df[[col]].notnull().sum()[0])

                # Get indices of non-null values
                not_null_indices = list(np.where(user_df[[col]].notnull())[0])

                # Shuffle to select random subset
                random.shuffle(not_null_indices)

                # Adjust indices to match full DataFrame index
                if user_df.index[0] != 0:
                    not_null_indices = [idx + user_df.index[0] for idx in not_null_indices]

                # Select indices to ampute
                amputed_indices = sorted(not_null_indices[:n_to_ampute])
                amputed_data_indices[user][col] = amputed_indices

                # Track counts and proportions
                amputed_count[user] += len(amputed_indices)
                amputed_avg[user] += len(amputed_indices) / len(col_values)

                # Apply amputation (set to NaN)
                df_to_ampute.loc[amputed_indices, col] = np.nan

            # Average proportion of amputations per user
            num_valid_columns = len(user_df.columns) - len(excluded_columns)
            amputed_avg[user] = amputed_avg[user] / num_valid_columns if num_valid_columns > 0 else 0

        # Summary statistics
        # print("\nAverage amputation ratios per user:")
        # print(amputed_avg)
        print(f"\nOverall mean amputed ratio: {np.mean(list(amputed_avg.values())):.4f}")

        return df_to_ampute, amputed_data_indices, amputed_count, amputed_avg
  
    def ampute_by_percentile_mnar_ii(self, lower_percentile: float, upper_percentile: float):
        """
        Amputes (sets to NaN) extreme values for each user's data based on percentile thresholds,
        creating a Missing Not At Random (MNAR) pattern focused on *tails* of the distribution.

        For each user and each feature:
            - Computes lower and upper percentile bounds.
            - Removes (NaNs) the data points that lie *outside* these bounds (i.e., extremes).
            - Returns the modified DataFrame and detailed statistics on amputations.

        Parameters
        ----------
        lower_percentile : float, optional
            The lower percentile threshold (default = 5).
        upper_percentile : float, optional
            The upper percentile threshold (default = 95).

        Returns
        -------
        df_to_ampute : pd.DataFrame
            The DataFrame with selected extreme values replaced by NaN.
        amputed_data_indices : dict
            Nested dictionary storing indices of amputed (NaN) values for each user and column.
        amputed_count : dict
            Dictionary mapping user → total number of amputed values.
        amputed_avg : dict
            Dictionary mapping user → average proportion of amputed values across columns.
        """

        # Columns that should not be modified
        excluded_columns = ['pid', 'date_map']

        # Copy the dataset for safe modification
        df = self.rapids_subset.copy()
        users = self.users
        df_to_ampute = df.copy()

        # Tracking structures
        amputed_data_indices = {}
        amputed_count = {}
        amputed_avg = {}

        print(f"Amputing data outside the {lower_percentile}th–{upper_percentile}th percentile range...")

        # Iterate through each user
        for user in tqdm.tqdm(users):
            user_df = df.loc[df['pid'] == user]
            amputed_data_indices[user] = {}
            amputed_count[user] = 0
            amputed_avg[user] = 0

            # Iterate through all columns for this user
            for col in user_df.columns:
                amputed_data_indices[user][col] = []

                if col in excluded_columns:
                    continue  # Skip identifier columns

                # Drop NaNs to compute percentiles
                col_values = user_df[col].dropna()
                if col_values.empty:
                    continue  # Skip if no valid data for this column

                # Compute percentile bounds for this column
                lower_bound = np.percentile(col_values, lower_percentile)
                upper_bound = np.percentile(col_values, upper_percentile)

                # Identify indices *outside* the percentile range (extreme values)
                mask = (user_df[col] < lower_bound) | (user_df[col] > upper_bound)
                ampute_indices = user_df.index[mask].tolist()

                # Store indices for tracking
                amputed_data_indices[user][col] = sorted(ampute_indices)

                # Update per-user stats
                amputed_count[user] += len(ampute_indices)
                amputed_avg[user] += len(ampute_indices) / len(col_values)

                # Set those values to NaN in the output DataFrame
                df_to_ampute.loc[ampute_indices, col] = np.nan

            # Average amputation ratio per user across columns
            num_valid_columns = len(user_df.columns) - len(excluded_columns)
            amputed_avg[user] = amputed_avg[user] / num_valid_columns if num_valid_columns > 0 else 0

        # Print summary statistics
        # print("\nAverage percentage of amputed data per user:")
        # print(amputed_avg)
        print(f"\nOverall mean amputed ratio: {np.mean(list(amputed_avg.values())):.4f}")

        return df_to_ampute, amputed_data_indices, amputed_count, amputed_avg


    
    def ampute_by_percentile_mnar_i(self, lower_percentile: float, upper_percentile: float):
        """
        Amputes (sets to NaN) values for each user's data based on percentile thresholds,
        creating a Missing Not At Random (MNAR) pattern.

        For each user and each feature:
            - Computes the lower and upper percentile bounds.
            - Removes (NaNs) the data points that lie *within* these bounds (i.e., middle range).
            - Returns the modified DataFrame and detailed stats on which values were amputed.

        Parameters
        ----------
        lower_percentile : float
            The lower percentile threshold (e.g., 40 means 40th percentile).
        upper_percentile : float
            The upper percentile threshold (e.g., 60 means 60th percentile).

        Returns
        -------
        df_to_ampute : pd.DataFrame
            The DataFrame with selected values replaced by NaN.
        amputed_data_indices : dict
            Nested dictionary storing indices of amputed values for each user and column.
        amputed_count : dict
            Dictionary mapping user → total number of amputed values.
        amputed_avg : dict
            Dictionary mapping user → average proportion of amputed values across columns.
        """

        # Columns that should not be modified
        excluded_columns = ['pid', 'date_map']

        # Copy the dataset for safe editing
        df = self.rapids_subset.copy()
        users = self.users
        df_to_ampute = df.copy()

        # Tracking dictionaries
        amputed_data_indices = {}
        amputed_count = {}
        amputed_avg = {}

        print(f"Amputing data between {lower_percentile}th and {upper_percentile}th percentiles...")

        # Iterate over all users
        for user in tqdm.tqdm(users):
            user_df = df.loc[df['pid'] == user]
            amputed_data_indices[user] = {}
            amputed_count[user] = 0
            amputed_avg[user] = 0

            # Iterate over all columns for this user
            for col in user_df.columns:
                amputed_data_indices[user][col] = []

                if col in excluded_columns:
                    continue  # Skip identifier columns

                # Drop NaN values to compute valid percentiles
                col_values = user_df[col].dropna()
                if col_values.empty:
                    continue  # Skip if no valid data for this user-column

                # Compute percentile bounds for this column
                lower_bound = np.percentile(col_values, lower_percentile)
                upper_bound = np.percentile(col_values, upper_percentile)

                # Identify rows *within* the middle percentile range
                # (You can invert mask to ampute high/low ends instead)
                mask = (user_df[col] > lower_bound) & (user_df[col] < upper_bound)
                ampute_indices = user_df.index[mask].tolist()

                # Store indices for tracking
                amputed_data_indices[user][col] = sorted(ampute_indices)

                # Update statistics
                amputed_count[user] += len(ampute_indices)
                amputed_avg[user] += len(ampute_indices) / len(col_values)

                # Set those values to NaN in the copy
                df_to_ampute.loc[ampute_indices, col] = np.nan

            # Compute average amputation ratio per column (excluding pid/date_map)
            num_valid_columns = len(user_df.columns) - len(excluded_columns)
            amputed_avg[user] = amputed_avg[user] / num_valid_columns if num_valid_columns > 0 else 0

        # Print summary statistics
        # print("\nAverage percentage of amputed data per user:")
        # print(amputed_avg)
        print(f"\nOverall mean amputed ratio: {np.mean(list(amputed_avg.values())):.4f}")

        return df_to_ampute, amputed_data_indices, amputed_count, amputed_avg

    
    def calculate_loss(self, imputations_dict={}):
        """
        Compute RMSE-based loss for each imputation method compared to the original dataset.

        Parameters
        ----------
        imputations_dict : dict
            Dictionary where keys are method names and values are DataFrames 
            containing imputed values.

        Returns
        -------
        loss : dict
            Nested dictionary of per-user losses for each method and baseline.
        """

        # Copy original and amputed datasets to avoid mutating class data
        original_df = self.rapids_subset.copy()
        amputed_data_indices = self.amputed_data_indices.copy()
        amputed_df = self.amputed_df.copy()
        amputed_count = self.amputed_count
        users = self.users

        loss = {}  # will hold user-wise loss summaries

        # Iterate over all users
        for user in tqdm.tqdm(users):
            loss[user] = {}

            # Extract per-user data
            user_df = original_df.loc[original_df['pid'] == user]
            amputed_user_df = amputed_df.loc[amputed_df['pid'] == user]

            # Initialize accumulators for all imputation methods and baselines
            method_losses = {name: 0 for name in imputations_dict.keys()}
            median_loss = mean_loss = median_28_loss = 0
            outlier_loss = nonoutlier_loss = 0
            count = outlier_count = nonoutlier_count = total_count = 0

            # Iterate through each feature column that has amputed indices
            for c in amputed_data_indices[user]:
                if c in ["pid", "date_map"]:
                    continue

                # Compute scaling and statistical values for normalization
                colmin, colmax = user_df[c].min(), user_df[c].max()
                colmedian = user_df[c].median(skipna=True)
                colmean = user_df[c].mean(skipna=True)
                col97_5 = user_df[c].quantile(0.975)
                col2_5 = user_df[c].quantile(0.025)

                diff = colmax - colmin
                if diff == 0:
                    continue  # skip constant columns

                # Iterate through all amputed (missing) indices for this column
                for k in amputed_data_indices[user][c]:
                    total_count += 1

                    # Compute baseline imputed values using the amputed dataframe
                    amputed_colmedian = amputed_user_df[c].median(skipna=True)
                    amputed_colmean = amputed_user_df[c].mean(skipna=True)
                    amputed_colmedian_28 = amputed_user_df[c][k - 28:k].median(skipna=True)
                    if np.isnan(amputed_colmedian_28):
                        amputed_colmedian_28 = 0

                    # Skip if baseline values are invalid (NaN)
                    if any(np.isnan([amputed_colmedian, amputed_colmedian_28, amputed_colmean])):
                        continue

                    # Skip if any imputed method's value is NaN for this position
                    if any(np.isnan(df[c][k]) for df in imputations_dict.values()):
                        continue

                    # Scale the original value between 0 and 1
                    scaled_original = (original_df[c][k] - colmin) / diff
                    if np.isnan(scaled_original):
                        continue

                    # --- Compute Loss for Each Imputation Method ---
                    for name, df in imputations_dict.items():
                        scaled_val = (df[c][k] - colmin) / diff
                        method_losses[name] += (scaled_original - scaled_val) ** 2

                    # --- Compute Baseline Losses ---
                    scaled_median = (amputed_colmedian - colmin) / diff
                    scaled_median_28 = (amputed_colmedian_28 - colmin) / diff
                    scaled_mean = (amputed_colmean - colmin) / diff

                    median_loss += (scaled_original - scaled_median) ** 2
                    median_28_loss += (scaled_original - scaled_median_28) ** 2
                    mean_loss += (scaled_original - scaled_mean) ** 2

                    count += 1

                    # --- Outlier Tracking ---
                    is_outlier = (original_df[c][k] >= col97_5 or original_df[c][k] <= col2_5)
                    if is_outlier:
                        outlier_count += 1
                        outlier_loss += (scaled_original - scaled_mean) ** 2
                    else:
                        nonoutlier_count += 1
                        nonoutlier_loss += (scaled_original - scaled_mean) ** 2

            # Prevent divide-by-zero errors
            count = max(count, 1)
            total_count = max(total_count, 1)
            outlier_count = max(outlier_count, 1)
            nonoutlier_count = max(nonoutlier_count, 1)

            # --- Compute Final RMSE Values ---
            for name in imputations_dict.keys():
                loss[user][name] = math.sqrt(method_losses[name] / count)

            # --- Add Baseline Metrics ---
            loss[user].update({
                "median": math.sqrt(median_loss / count),
                "median_28": math.sqrt(median_28_loss / count),
                "mean": math.sqrt(mean_loss / count),
                "outlier": math.sqrt(outlier_loss / outlier_count),
                "nonoutlier": math.sqrt(nonoutlier_loss / nonoutlier_count),
                "imputed_count": count,
                "total_count": total_count,
                "amputed_count": amputed_count[user],
                "nonoutlier_count": nonoutlier_count,
                "outlier_count": outlier_count,
            })

        return loss
    
    def display_loss(self, loss, types, remove_outliers=False, percentile_removed=0.05):
        """
        Visualize loss data for different metric types, with optional outlier removal.

        Args:
            loss (dict): Dictionary containing loss values for each user. 
                         Expected structure: loss[user][type] and loss[user]["data%"].
            types (list[str]): List of metric types to plot (e.g., ["lstm", "moment", "top"]).
            remove_outliers (bool, optional): If True, removes extreme values based on percentile thresholds.
            percentile_removed (float, optional): Fraction (0–0.5) of data to remove from each tail when remove_outliers=True. 
            Default is 0.05 (removes 5% from each side).

        Returns:
            matplotlib.pyplot: The matplotlib.pyplot object with the plotted data.
        """

        y_removed = 0
        plt.figure(figsize=(8, 6))

        print("Number of users in loss file:", len(loss.keys()))

        # Aggregate losses per user
        temp = {}
        for user in loss:
            if user not in temp:
                temp[user] = {}
            for t in types:
                temp[user][t] = temp[user].get(t, 0) + loss[user][t]
            temp[user]["data%"] = loss[user]["data%"]

        # Plot each type
        for key in types:
            x, y = [], []

            for user in temp.keys():
                x.append(temp[user]["data%"])
                y.append(temp[user][key])

            if remove_outliers:
                y = np.array(y)
                x = np.array(x)

                lower_q = np.quantile(y, percentile_removed)
                upper_q = np.quantile(y, 1.0 - percentile_removed)

                mask = (y >= lower_q) & (y <= upper_q)
                removed = len(y) - np.sum(mask)

                y_removed += removed
                x = x[mask]
                y = y[mask]

                print(f"Removed {removed} outliers for type '{key}'")

            plt.scatter(x, y, label=key)

        plt.xlabel("Data Availability (%)")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        
        return plt
        
        



