import os
import math
import random
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from fancyimpute import SoftImpute
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.regularizers import l1_l2

import momentfm
from momentfm import MOMENTPipeline

from collections import defaultdict

import parameters as parameters
from utils import Utils

# Suppress specific warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

class Imputation:
    """
    A class for handling data imputation operations on RAPIDS datasets.
    """

    def __init__(self, rapids_df, rapids_subset, users=[], index_to_dates={}, algo="dl_feat"):
        """
        Initialize the imputation class with dataset and configuration.

        Parameters
        ----------
        rapids_df : pd.DataFrame
            Full RAPIDS dataset.
        rapids_subset : pd.DataFrame
            Subset of RAPIDS dataset with selected features.
        users : list
            List of user IDs.
        index_to_dates : dict
            Mapping from index → date.
        algo : str
            Algorithm key used to pick feature subset from `parameters.algo_dict`.
        """
        self.rapids_df = rapids_df
        self.rapids_subset = rapids_subset
        self.users = users
        self.index_to_dates = index_to_dates
        self.algo_dict = parameters.algo_dict
        self.algo = algo
        self.columns = list(self.algo_dict[self.algo])


    def compute_similarity(self):
        """
        Computes similarity between all days for each user based on normalized
        feature values. Returns a dictionary of similarity matrices keyed by user.
        """
        similarity_matrices = {}
        excluded_columns = ['pid', 'date_map']
        rapids_df = self.rapids_df.copy()

        for user in tqdm.tqdm(self.users):
            user_df = rapids_df.loc[rapids_df.pid == user].dropna(how="all", axis=1)

            # Remove identifier columns and set index as date_map
            user_df = user_df.drop(["pid"], axis=1)
            user_df.set_index(user_df["date_map"], inplace=True)

            # Select numeric columns for similarity computation
            num_cols = user_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'date_map' in num_cols:
                num_cols.remove('date_map')

            # Clip extreme values to 5th–95th percentile
            user_df = user_df[num_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

            # Normalize feature values to [0, 1]
            user_df = (user_df - user_df.min()) / (user_df.max() - user_df.min())

            # Initialize empty similarity matrix (date_map × date_map)
            n_dates = len(self.index_to_dates)
            similarity_matrix = [[0 for _ in range(n_dates)] for _ in range(n_dates)]

            # Compute pairwise similarity between all days
            for i, row1 in user_df.iterrows():
                for j, row2 in user_df.iterrows():
                    diff = []
                    for c in user_df.columns:
                        if c not in excluded_columns and not pd.isnull(row1[c]) and not pd.isnull(row2[c]):
                            diff.append((row1[c] - row2[c]) ** 2)
                    if diff:
                        similarity_matrix[i][j] = 1 - math.sqrt(sum(diff) / len(diff))

            similarity_matrices[user] = similarity_matrix

        return similarity_matrices


# does imputation based on within user similarity in 3 ways: random, weekly, and top
    def bounded_knn(self, similarity_type="top", min_count=2, max_count=6):
        """
        Imputes missing values based on within-user similarity in three ways:
        1. Random: Randomly select `k` similar days.
        2. Weekly: Select days matching the same weekday pattern.
        3. Top: Select `k` most similar days by similarity score.

        Parameters
        ----------
        min_count : int
            Minimum number of valid days required to perform imputation.
        max_count : int
            Maximum number of days (K) to consider for weighted averaging.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            DataFrames imputed using top, random, and weekly similarity strategies.
        """
        similarity_matrices = self.compute_similarity()
        df = self.rapids_df.copy()
        columns = self.columns.copy()

        # Store available data for each user and feature by date
        user_data_for_dates = defaultdict(lambda: defaultdict(list))
        for _, row in df.iterrows():
            for c in columns:
                if not pd.isna(row[c]):
                    user_data_for_dates[row['pid']][c].append((row["date_map"], row[c]))

        # Prepare output DataFrames
        
        if similarity_type=="random":
            random_similarity_df = df.copy()
        elif similarity_type=="top":
            top_similarity_df = df.copy()
        elif similarity_type=="weekly":
            weekly_similarity_df = df.copy()

        for i, row in tqdm.tqdm(df.iterrows()):
            user = row['pid']
            date = row['date_map']
            similarity_matrix = similarity_matrices[user]

            for c in columns:
                if pd.isna(row[c]):
                    days_having_data = user_data_for_dates[user][c]
                    if len(days_having_data) < min_count:
                        continue

                    k = min(len(days_having_data), max_count)

                    # ----- Random Similarity -----
                    if similarity_type=="random":
                        random_days = random.sample(days_having_data, k)
                        random_coeffs = [similarity_matrix[date][d] for d, _ in random_days]
                        random_vals = [v for _, v in random_days]

                        if sum(random_coeffs) != 0:
                            imputed_value = sum(a * b for a, b in zip(random_vals, random_coeffs)) / sum(random_coeffs)
                            random_similarity_df.at[i, c] = imputed_value

                    # ----- Top-K Similarity -----
                    elif similarity_type=="top":
                        top_similarity = sorted(
                            [(d, v, similarity_matrix[date][d]) for d, v in days_having_data],
                            key=lambda x: x[2],
                            reverse=True
                        )[:k]

                        top_coeffs = [s[2] for s in top_similarity]
                        top_vals = [s[1] for s in top_similarity]

                        if sum(top_coeffs) != 0:
                            imputed_value = sum(a * b for a, b in zip(top_vals, top_coeffs)) / sum(top_coeffs)
                            top_similarity_df.at[i, c] = imputed_value

                    # ----- Weekly Similarity -----
                    elif similarity_type=="weekly":
                        weekly_days = [(d, v) for d, v in days_having_data if d % 7 == date % 7]
                        weekly_coeffs = [similarity_matrix[date][d] for d, _ in weekly_days]
                        weekly_vals = [v for _, v in weekly_days]

                        if sum(weekly_coeffs) != 0:
                            imputed_value = sum(a * b for a, b in zip(weekly_vals, weekly_coeffs)) / sum(weekly_coeffs)
                            weekly_similarity_df.at[i, c] = imputed_value
                            
        if similarity_type=="random":
            return random_similarity_df
        elif similarity_type=="top":
            return top_similarity_df
        elif similarity_type=="weekly":
            return weekly_similarity_df

    def simple_knn(self, k=6, across_users=False):
        """
        Performs KNN imputation either per user or across all users.

        Parameters
        ----------
        k : int
            Number of nearest neighbors.
        across_users : bool
            If True, perform a second KNN imputation across all users.

        Returns
        -------
        pd.DataFrame
            Imputed RAPIDS dataset.
        """
        columns_to_impute = self.columns.copy()
        rapids_df = self.rapids_df.copy()
        users = self.users
        imputed_dfs = []

        imputer = KNNImputer(n_neighbors=k, weights='distance', keep_empty_features=False)

        for user in tqdm.tqdm(users):
            user_df = rapids_df.loc[rapids_df.pid == user].copy()
            user_df_subset = user_df[columns_to_impute]

            # Store and drop all-null columns
            dropped_columns = list(user_df_subset.columns[user_df_subset.isna().all()])
            user_df_subset = user_df_subset.dropna(how="all", axis=1)
            columns_to_impute_user = [col for col in columns_to_impute if col not in dropped_columns]

            # Preserve feature ranges
            min_vals = user_df_subset.min()
            max_vals = user_df_subset.max()

            # Perform imputation
            user_df_imputed = imputer.fit_transform(user_df_subset)
            user_df_imputed = pd.DataFrame(user_df_imputed, columns=columns_to_impute_user)

            # Clip values to original range
            for col in user_df_imputed.columns:
                user_df_imputed[col] = np.clip(user_df_imputed[col], min_vals[col], max_vals[col])

            # Replace missing columns with imputed values
            user_df.reset_index(drop=True, inplace=True)
            user_df[columns_to_impute_user] = user_df_imputed

            imputed_dfs.append(user_df)

        # Combine all user DataFrames
        imputed_rapids_df = pd.concat(imputed_dfs, ignore_index=True)

        # Optional: Run another KNN imputation across all users
        if across_users:
            subset = imputed_rapids_df[columns_to_impute]
            imputer = KNNImputer(n_neighbors=k, weights='distance', keep_empty_features=True)
            subset_imputed = imputer.fit_transform(subset)
            imputed_rapids_df[columns_to_impute] = subset_imputed

        return imputed_rapids_df
    
    
    def mice(self, across_users=False, is_knn=False):
        """
        Perform Multiple Imputation by Chained Equations (MICE) for each user.

        Parameters
        ----------
        across_users : bool, optional
            If True, performs an additional imputation across all users (disabled here).
        is_knn : bool, optional
            If True, could use KNNImputer for initial imputation (currently commented).

        Returns
        -------
        pd.DataFrame
            The imputed RAPIDS dataset.
        """
        columns_to_impute = self.columns.copy()
        rapids_df = self.rapids_df.copy()
        users = self.users
        imputed_dfs = []

        imputer = IterativeImputer(max_iter=10, random_state=4)

        for user in tqdm.tqdm(users):
            user_df = rapids_df.loc[rapids_df.pid == user].copy()

            # Select only the relevant columns
            user_df_subset = user_df[columns_to_impute]

            # Drop all-null columns
            dropped_columns = list(user_df_subset.columns[user_df_subset.isna().all()])
            user_df_subset = user_df_subset.dropna(how="all", axis=1)
            columns_to_impute_user = [col for col in columns_to_impute if col not in dropped_columns]

            # Record feature min/max (not currently used for clipping)
            min_vals = user_df_subset.min()
            max_vals = user_df_subset.max()

            user_df_subset = user_df_subset[columns_to_impute_user]

            try:
                # Apply IterativeImputer
                user_df_imputed = imputer.fit_transform(user_df_subset)
                user_df_imputed = pd.DataFrame(user_df_imputed, columns=columns_to_impute_user)

                # Replace imputed values in the original DataFrame
                user_df.reset_index(drop=True, inplace=True)
                user_df[columns_to_impute_user] = user_df_imputed

            except ValueError as e:
                # Handle cases where imputation fails (e.g., insufficient data)
                user_df.reset_index(drop=True, inplace=True)
                print(f"Imputation failed for user {user} with error: {e}")

            imputed_dfs.append(user_df)

        # Combine all user DataFrames
        imputed_rapids_df = pd.concat(imputed_dfs, ignore_index=True)
        
        return imputed_rapids_df
    

    def autoencoder(self, layers="two", min_max=True, use_distribution=False, temporary_imputation="knn"):
        """
        Impute missing values using an autoencoder model trained per user.

        Parameters
        ----------
        layers : str
            Type of autoencoder architecture: "one", "two", or "sigmoid".
        min_max : bool
            Whether to apply min-max scaling to features.
        use_distribution : bool
            Whether to model each feature with Gaussian Mixture Models.
        temporary_imputation : str
            Strategy for temporary filling of NaNs before training ("median" or "knn").

        Returns
        -------
        pd.DataFrame
            Autoencoder-imputed dataset.
        """
        imputed_dfs = []
        torch.manual_seed(42)

        cols = self.columns.copy()
        df = self.rapids_df.copy()
        users = self.users

        for user in tqdm.tqdm(users):
            imputer = KNNImputer(n_neighbors=6, weights='distance', keep_empty_features=False)
            columns_to_impute = self.columns.copy()

            user_df = df.loc[df.pid == user].copy()
            user_df.reset_index(drop=True, inplace=True)
            user_df_subset = user_df[columns_to_impute]

            # Drop columns that are entirely NaN
            dropped_columns = list(user_df_subset.columns[user_df_subset.isna().all()])
            user_df_subset = user_df_subset.dropna(how="all", axis=1)
            columns_to_impute_user = [col for col in columns_to_impute if col not in dropped_columns]
            user_df_subset = user_df_subset[columns_to_impute_user]

            # Apply Min-Max scaling (optional)
            if min_max:
                maxm = user_df_subset.max()
                minm = user_df_subset.min()
                difference = (maxm - minm)
                difference.replace(0, -1, inplace=True)
                user_df_subset = (user_df_subset - minm) / difference
                difference.replace(-1, 0, inplace=True)

            tensor_data = torch.tensor(user_df_subset.values, dtype=torch.float32)

            # Mask indicates missing values
            mask = torch.isnan(tensor_data)

            # Temporary imputation before training
            if temporary_imputation == "median":
                temporary = torch.tensor(user_df_subset.median().values, dtype=torch.float32)
            elif temporary_imputation == "knn":
                temp_df = pd.DataFrame(
                    imputer.fit_transform(user_df_subset),
                    columns=columns_to_impute_user
                )
                temporary = torch.tensor(temp_df.values, dtype=torch.float32)

            # Optionally model feature distributions with Gaussian Mixtures
            distributions = []
            if use_distribution:
                for column in columns_to_impute_user:
                    gmm = GaussianMixture(n_components=2)
                    col_value = user_df_subset[[column]].dropna()
                    if len(col_value) < 5:
                        distributions.append(None)
                        continue
                    gmm.fit(col_value)
                    distributions.append(gmm)

            # Fill temporary values into missing entries
            if temporary_imputation == "median":
                for i in range(tensor_data.shape[1]):
                    tensor_data[mask[:, i], i] = temporary[i]
            elif temporary_imputation == "knn":
                tensor_data[mask] = temporary[mask]

            dataset = TensorDataset(tensor_data, mask)
            data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

            input_size = tensor_data.shape[1]
            hidden_size = 20
            encoding_size = 10

            # Adjust learning rate based on scaling and architecture
            if min_max:
                learning_rate = 0.001 if layers == "sigmoid" else 0.0001
            else:
                learning_rate = 0.001

            num_epochs = 10
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Select autoencoder architecture
            if layers == "two":
                autoencoder = AutoencoderTwoLayer(input_size, hidden_size, encoding_size).to(device)
            elif layers == "one":
                autoencoder = AutoencoderOneLayer(input_size, hidden_size).to(device)
            elif layers == "sigmoid":
                autoencoder = AutoencoderOneLayerWithSigmoid(input_size, hidden_size).to(device)
            else:
                print("Invalid layer type. Using two-layer autoencoder.")
                autoencoder = AutoencoderTwoLayer(input_size, hidden_size, encoding_size).to(device)

            custom_loss = CustomLoss()
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

            best_loss = float('inf')
            best_epoch = -1
            best_autoencoder_state = None

            # ---------------- Training Loop ----------------
            for epoch in range(num_epochs):
                for batch_data, batch_mask in data_loader:
                    batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
                    output = autoencoder(batch_data)
                    loss = custom_loss(output, batch_data, batch_mask, distributions)
                    state_dict = autoencoder.state_dict()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_autoencoder_state = copy.deepcopy(state_dict)

            # Load best weights and reconstruct
            if best_autoencoder_state:
                autoencoder.load_state_dict(best_autoencoder_state)
            autoencoder.eval()

            with torch.no_grad():
                reconstructed_data = autoencoder(tensor_data.to(device)).cpu()
                tensor_data[mask] = reconstructed_data[mask]

            # Convert back to DataFrame
            imputed_df = pd.DataFrame(tensor_data.numpy(), columns=columns_to_impute_user)

            # Fallback: if reconstruction fails, use KNN
            if torch.isnan(reconstructed_data).all():
                imputed_df = pd.DataFrame(
                    imputer.fit_transform(user_df_subset),
                    columns=columns_to_impute_user
                )
                print("Using KNN fallback.")

            # Reverse scaling
            if min_max:
                imputed_df = (imputed_df * difference) + minm

            # Replace in original DataFrame
            user_df[columns_to_impute_user] = imputed_df
            imputed_dfs.append(user_df)

        # Combine all imputed DataFrames
        imputed_rapids_df = pd.concat(imputed_dfs, ignore_index=True)
        
        return imputed_rapids_df
    
    def lstm(self):
        """
        Perform imputation using an LSTM model trained per user.

        The model reconstructs missing sequences using temporal dependencies.
        Falls back to KNN if training fails.
        """
        imputed_df_list = []
        columns_to_impute = self.columns.copy()
        rapids_df = self.rapids_df.copy()
        users = self.users

        for user in tqdm.tqdm(users):
            imputer = KNNImputer(n_neighbors=6, weights='distance', keep_empty_features=False)
            user_df = rapids_df.loc[rapids_df.pid == user].copy()
            user_df.reset_index(drop=True, inplace=True)

            user_df_subset = user_df[columns_to_impute]
            dropped_columns = list(user_df_subset.columns[user_df_subset.isna().all()])
            user_df_subset = user_df_subset.dropna(how="all", axis=1)
            columns_to_impute_user = [col for col in columns_to_impute if col not in dropped_columns]
            user_df_subset = user_df_subset[columns_to_impute_user]
            df_original = user_df_subset.copy()

            # Skip users with no data
            if df_original.isnull().all().all():
                continue

            # Temporary KNN fill
            df_temp_imputed = imputer.fit_transform(df_original)
            df_temp_imputed = pd.DataFrame(df_temp_imputed, columns=columns_to_impute_user)

            # Create mask for missing values
            mask = df_original.isnull().astype(float).values

            # Scale data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df_temp_imputed)

            input_size = len(columns_to_impute_user)
            hidden_size = 20
            seq_length = 7
            batch_size = 1
            num_epochs = 10
            learning_rate = 0.001

            # Prepare dataset
            dataset = TimeSeriesDataset(data_scaled, mask, seq_length)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Initialize model and optimizer
            model = LSTMImputer(input_size, hidden_size)
            custom_loss = CustomLossLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            best_epoch = -1
            best_model_state = None

            # ---------------- Training Loop ----------------
            for epoch in range(num_epochs):
                total_loss = 0
                for sequences, targets, masks in dataloader:
                    outputs = model(sequences)
                    loss = custom_loss(outputs, targets, masks)
                    state_dict = model.state_dict()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(state_dict)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

            # Load best model and perform imputation
            if best_model_state:
                model.load_state_dict(best_model_state)

            imputed_user_df = self.impute_missing_values(
                df_temp_imputed, df_original, model, seq_length, scaler
            )

            # Fallback: KNN if LSTM output invalid
            if imputed_user_df.isnull().all().all():
                imputed_user_df = pd.DataFrame(
                    imputer.fit_transform(df_original),
                    columns=columns_to_impute_user
                )
                print("Using KNN fallback.")

            user_df[columns_to_impute_user] = imputed_user_df
            imputed_df_list.append(user_df)

        # Combine all user DataFrames
        imputed_dfs = pd.concat(imputed_df_list, ignore_index=True)
        
        return imputed_dfs
    
    def median_28(self):
        """
        Impute missing values using a rolling 28-day median for each user.
        For each missing value in a user's data, replaces it with the median of
        the previous 28 available entries in the same column.
        """
        imputed_dfs = []
        rapids_df = self.rapids_df.copy()
        rapids_subset = self.rapids_subset.copy()
        users = self.users
        columns = self.columns.copy()

        for user in tqdm.tqdm(users):
            user_df = rapids_df.loc[rapids_df.pid == user]
            user_subset = rapids_subset.loc[rapids_subset.pid == user].copy()
            user_imputed = user_subset.copy()

            # Iterate through each row and column for imputation
            for i, row in user_subset.iterrows():
                for c in columns:
                    if pd.isna(row[c]):
                        # Compute median of last 28 values in column
                        median_28 = user_subset[c][i - 28:i].median(skipna=True)
                        if np.isnan(median_28):
                            median_28 = 0
                        user_imputed.at[i, c] = median_28

            # Replace the original columns with imputed ones
            user_df[columns] = user_imputed
            imputed_dfs.append(user_df)

        # Combine all user DataFrames
        imputed_rapids_df = pd.concat(imputed_dfs, ignore_index=True)
        
        return imputed_rapids_df

    def moment(self):
        """
        Impute missing values using the MOMENT deep learning model for time-series data.
        Includes preprocessing (scaling, temporary imputation), model training, and reconstruction.
        """
        imputed_dfs = []
        rapids_df = self.rapids_df.copy()
        users = self.users
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for user in tqdm.tqdm(users):
            imputer = KNNImputer(n_neighbors=6, weights='distance')
            columns_to_impute = self.columns.copy()
            user_df = rapids_df.loc[rapids_df.pid == user].copy()
            user_df.reset_index(drop=True, inplace=True)
            user_subset = user_df[columns_to_impute].copy()

            # Choose sequence length based on available data
            multipletotake = [88, 80, 72]
            mtt = next((m for m in multipletotake if len(user_subset) > m), len(user_subset))

            # Split into initial segment and leftover
            leftover = user_subset.tail(len(user_subset) - mtt)
            user_subset = user_subset.head(mtt)

            # Drop all-null columns
            dropped_cols = user_subset.columns[user_subset.isna().all()]
            user_subset = user_subset.drop(columns=dropped_cols)
            columns_to_impute_user = [col for col in columns_to_impute if col not in dropped_cols]
            user_subset = user_subset[columns_to_impute_user]
            leftover = leftover[columns_to_impute_user]

            # Min-max scaling
            min_max = True
            if min_max:
                max_vals = user_subset.max()
                min_vals = user_subset.min()
                diff = (max_vals - min_vals).replace(0, -1)
                user_subset = (user_subset - min_vals) / diff
                diff.replace(-1, 0, inplace=True)

            # Convert to tensor and handle NaNs
            tensor_data = torch.tensor(user_subset.values, dtype=torch.float32)
            mask = torch.isnan(tensor_data)

            # Temporary imputation using KNN
            if temporary_imputation=="median":
                temporary = torch.tensor(user_df_subset.median().values, dtype=torch.float32)
                for i in range(tensor_data.shape[1]):
                    tensor_data[tensor_data[:, i], i] = temporary[i]

            elif temporary_imputation == "knn":
                temporary = imputer.fit_transform(user_subset)
                temporary = torch.tensor(temporary, dtype=torch.float32)
                tensor_data[mask] = temporary[mask]

            # Build dataset and dataloader
            seq_length = 8
            dataset = TimeSeriesDataset(tensor_data, mask, seq_length)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

            # Initialize MOMENT model
            model = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large",
                model_kwargs={'task_name': 'reconstruction'}
            )
            model.init()

            # Define optimizer and loss
            custom_loss = CustomLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            best_loss = float('inf')
            best_state = None
            for epoch in range(10):
                total_loss = 0
                for sequences, targets, masks in dataloader:
                    n_channels = sequences.shape[2]
                    sequences = sequences.permute(0, 2, 1)
                    masks = masks.to(device).long().permute(0, 2, 1).reshape((-1, 1, seq_length))
                    if masks.shape[0] < n_channels * seq_length:
                        continue

                    # Forward pass
                    outputs = model(x_enc=sequences)
                    reconstruction = outputs.reconstruction.reshape((-1, 1, seq_length))
                    sequences = sequences.reshape((-1, 1, seq_length))

                    # Compute loss
                    loss = custom_loss(reconstruction, sequences, masks, [])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Track best model
                if loss < best_loss:
                    best_loss = loss
                    best_state = copy.deepcopy(model.state_dict())

                print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

            # Load best model state
            if best_state:
                model.load_state_dict(best_state)
            model.eval()

            # Reconstruct data
            with torch.no_grad():
                tensor_data = tensor_data.T.unsqueeze(0)
                reconstructed = model(x_enc=tensor_data).reconstruction.squeeze(0).T
                tensor_data = tensor_data.squeeze(0).T
                tensor_data[mask] = reconstructed[mask]

            # Convert to DataFrame and reverse scaling
            imputed_user_df = pd.DataFrame(tensor_data.numpy(), columns=columns_to_impute_user)
            if torch.isnan(reconstructed).all():
                imputed_user_df = imputer.fit_transform(user_subset)
                imputed_user_df = pd.DataFrame(imputed_user_df, columns=columns_to_impute_user)

            if min_max:
                imputed_user_df = (imputed_user_df * diff) + min_vals

            user_df[columns_to_impute_user] = imputed_user_df
            imputed_dfs.append(user_df)

        return pd.concat(imputed_dfs, ignore_index=True)
    
    def matrix_completion(self):
        """
        Perform matrix completion imputation using SoftImpute for each user.
        Normalizes data, performs low-rank matrix completion, and restores scaling.
        """
        imputed_df = pd.DataFrame()
        df = self.rapids_df.copy()
        users = self.users

        for user in tqdm.tqdm(users):
            user_df = df.loc[df.pid == user].copy().reset_index(drop=True)
            full_user_df = user_df.copy()

            # Drop non-numeric and all-null columns
            user_df = user_df.dropna(how="all", axis=1).drop(columns=["pid"])
            num_cols = user_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'date_map' in num_cols:
                num_cols.remove('date_map')

            # Clip outliers and standardize
            user_df = user_df[num_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))
            mean, std = user_df.mean(), user_df.std()
            user_df = (user_df - mean) / std
            user_df = user_df.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)

            # Skip if all data missing
            if user_df.isnull().all().all():
                imputed_df = pd.concat([imputed_df, user_df], axis=0)
                continue

            # Apply SoftImpute
            imputed_user_df = pd.DataFrame(SoftImpute(verbose=False).fit_transform(user_df), columns=user_df.columns)
            imputed_user_df = (imputed_user_df * std) + mean

            # Insert imputed values into full DataFrame
            for col in user_df.columns:
                full_user_df[col] = imputed_user_df[col]

            imputed_df = pd.concat([imputed_df, full_user_df], axis=0)

        return imputed_df.reset_index(drop=True)

    
    def masked_loss_function(self, predictions, targets, mask):
        """
        Computes Mean Squared Error (MSE) loss only on the *non-missing* (observed) values.
        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
            mask (array-like): Boolean mask indicating missing (True) and present (False) values.
        Returns:
            torch.Tensor: Scalar loss value (mean over observed entries).
        """
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        inverted_mask = ~mask_tensor  # Selects present (non-missing) values
        loss = ((predictions - targets) ** 2) * inverted_mask

        if inverted_mask.sum() > 0:
            return loss.sum() / inverted_mask.sum()  # Normalize by count of present values
        else:
            return np.nan  # Handle case where no valid entries exist

    
    def impute_missing_values(self, df_temp_imputed, df_original, model, seq_length, scaler):
        """
        Iteratively imputes missing values in a DataFrame using a trained model.
        Args:
            df_temp_imputed (pd.DataFrame): Initially imputed data (temporary).
            df_original (pd.DataFrame): Original data containing NaNs.
            model (nn.Module): Trained imputation model.
            seq_length (int): Sequence length for time-series model.
            scaler (object): Scaler used for normalization.
        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        df_copy = df_temp_imputed.copy()
        model.eval()
        data_scaled = scaler.transform(df_copy)

        with torch.no_grad():
            for i in range(seq_length, len(df_copy)):
                seq = data_scaled[i - seq_length:i]
                seq_tensor = torch.Tensor(seq).unsqueeze(0)
                imputed_values = model(seq_tensor).numpy()[0]

                for j in range(len(df_copy.columns)):
                    if pd.isnull(df_original.iloc[i, j]):
                        df_copy.iloc[i, j] = scaler.inverse_transform([imputed_values])[0][j]

        return df_copy

class CustomLoss_timeseries(nn.Module):
    """
    Custom loss for time series imputation.
    Computes MSE only on observed values (mask=False).
    """
    def __init__(self):
        super(CustomLoss_timeseries, self).__init__()

    def forward(self, predicted, target, mask):
        alpha = 0.2
        distribution_loss = 0.0
        inverted_mask = ~mask
        loss = ((predicted - target) ** 2 * inverted_mask).sum() / inverted_mask.sum()
        
        if inverted_mask.sum() == 0 or torch.isnan(loss):
            return torch.tensor(100.0, requires_grad=True)
        return loss + alpha * distribution_loss

    

class CustomLoss(nn.Module):
    """
    Custom loss combining reconstruction loss (MSE) and distribution-based regularization.
    Used for MOMENT-based or probabilistic imputation models.
    """
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target, mask, distributions):
        alpha = 0.2
        inverted_mask = ~mask
        loss = ((predicted - target) ** 2 * inverted_mask).sum() / inverted_mask.sum()

        distribution_loss = 0.0
        if distributions:
            results = []
            probs = predicted.clone().detach()

            for i, gmm in enumerate(distributions):
                feature = probs[:, i].unsqueeze(1)
                if gmm:
                    feature = torch.nan_to_num(feature)
                    log_prob = -torch.tensor(gmm.score_samples(feature)).unsqueeze(1)
                else:
                    log_prob = torch.zeros_like(feature)
                results.append(log_prob)

            results = torch.stack(results, dim=1).squeeze(dim=2)
            distribution_loss = (results * mask).sum() / mask.sum()

        return loss + alpha * distribution_loss


class LSTMImputer(nn.Module):
    """
    Simple LSTM-based imputer for sequential data.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMImputer, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # Output at last timestep


    
class CustomLossLSTM(nn.Module):
    """
    Custom loss function for LSTM imputer.
    """
    def __init__(self):
        super(CustomLossLSTM, self).__init__()

    def forward(self, predicted, target, mask):
        alpha = 0.
        distribution_loss = 0.0
        inverted_mask = ~mask
        loss = ((predicted - target) ** 2 * inverted_mask).sum() / inverted_mask.sum()

        if inverted_mask.sum() == 0 or torch.isnan(loss):
            return torch.tensor(100.0, requires_grad=True)
        return loss + alpha * distribution_loss


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset class for sequential (time-series) imputation models.
    Produces (sequence, target, mask) tuples for training.
    """
    def __init__(self, data, mask, seq_length=7):
        self.data = data
        self.mask = mask
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+self.seq_length]
        mask_seq = self.mask[idx:idx+self.seq_length]
        return torch.Tensor(seq), torch.tensor(target, dtype=torch.float32), torch.tensor(mask_seq, dtype=torch.bool)

class AutoencoderTwoLayer(nn.Module):
    """Two-layer Autoencoder for imputation/reconstruction tasks."""
    def __init__(self, input_size, hidden_size, encoding_size):
        super(AutoencoderTwoLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, encoding_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    
class AutoencoderOneLayerWithSigmoid(nn.Module):
    """Single-layer Autoencoder with Sigmoid activation in decoder."""
    def __init__(self, input_size, hidden_size):
        super(AutoencoderOneLayerWithSigmoid, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoencoderOneLayer(nn.Module):
    """Single-layer Autoencoder (Linear decoder, no activation)."""
    def __init__(self, input_size, hidden_size):
        super(AutoencoderOneLayer, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size))

    def forward(self, x):
        return self.decoder(self.encoder(x))