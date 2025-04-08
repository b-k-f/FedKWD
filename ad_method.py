import pandas as pd
import numpy as np
import joblib
import time

import torch
import torch.nn as nn
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,  precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    average_precision_score, mean_squared_error, mean_absolute_error, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score, homogeneity_score, completeness_score, v_measure_score

def flatten_weights(state_dict):
    weights = []
    for key in state_dict:
        weights.append(state_dict[key].cpu().flatten())  # Move tensor to CPU before flattening
    return torch.cat(weights)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomalies_with_autoencoder(model, threshold):
    criterion = nn.MSELoss()
    num_ep = 3
    btch_size = 32
    initial_flattened_weights = flatten_weights(model.state_dict()).numpy().reshape(1, -1)
    ds = torch.tensor(initial_flattened_weights, dtype=torch.float32)
    inpt_dm = ds.shape[1]
    autoencoder = Autoencoder(inpt_dm)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=btch_size, shuffle=True)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

    for epoch in range(num_ep):
        for data in data_loader:
            output = autoencoder(data)
            lss = criterion(output, data)

            optimizer.zero_grad()
            lss.backward()
            optimizer.step()
            
    reconstruction_error = torch.mean((ds - output) ** 2).item()
    is_anomaly = reconstruction_error > threshold
    return is_anomaly, reconstruction_error

def test_scores(y_test, y_predictions, x_test, ad_method, prct):
    # test scores
    acc = accuracy_score(y_test, y_predictions)
    conf = confusion_matrix(y_test, y_predictions)
    precision = precision_score(y_test, y_predictions,)
    recall = recall_score(y_test, y_predictions)
    f1 = f1_score(y_test, y_predictions)
    roc_auc = roc_auc_score(y_test, y_predictions)
    avg_precision = average_precision_score(y_test, y_predictions)
    mse = mean_squared_error(y_test, y_predictions)
    mae = mean_absolute_error(y_test, y_predictions)
    ari = adjusted_rand_score(y_test, y_predictions)
    homogeneity = homogeneity_score(y_test, y_predictions)
    completeness = completeness_score(y_test, y_predictions)
    v_measure = v_measure_score(y_test, y_predictions)

    fpr, tpr, _ = roc_curve(y_test, y_predictions)
    
    print(' Anomaly detection method: ', ad_method)
    print("Accuracy:", acc)
    print("Confusion Matrix:", conf)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print("Average Precision:", avg_precision)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Adjusted Rand Index:", ari)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    
    # silhouette = silhouette_score(x_test, y_predictions)
    davies_bouldin = davies_bouldin_score(x_test, y_predictions)
    calinski_harabasz = calinski_harabasz_score(x_test, y_predictions)
    # print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", davies_bouldin)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    
    ad_metrics = {
    'Accuracy': acc,
    'Confusion Matrix': conf,
    'Precision': avg_precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': roc_auc,    
    'FPR': fpr,
    'TPR': tpr,
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Adjusted Rand Index': ari,
    'Homogeneity': homogeneity,
    'Completeness': completeness,
    'V-measure': v_measure,
    # 'Silhouette Score': silhouette,
    'Davies-Bouldin Index': davies_bouldin,
    'Calinski-Harabasz Index': calinski_harabasz,
    }
    return ad_metrics

def impute_outliers(X_train, X_test, x_predictions, y_predictions):
    X_train.reset_index(inplace = True)
    X_test.reset_index(inplace = True)
    # Copy the data into NumPy arrays for faster computation
    X_train_arr = X_train.values
    X_test_arr = X_test.values

    # Impute outliers in the training set
    x_outlier_indices = np.where(x_predictions == -1)[0]
    x_good_indices = np.where(x_predictions == 1)[0]
    for outlier_index in x_outlier_indices:
        try:
            previous_good_index = x_good_indices[x_good_indices < outlier_index][-1]
        except IndexError:
            previous_good_index = x_good_indices[outlier_index]
        # Replace the outlier value with the previous good value
        X_train_arr[outlier_index] = X_train_arr[previous_good_index]

    # Impute outliers in the test set
    y_outlier_indices = np.where(y_predictions == -1)[0]
    y_good_indices = np.where(y_predictions == 1)[0]
    for outlier_index in y_outlier_indices:
        try:
            previous_good_index = y_good_indices[y_good_indices < outlier_index][-1]
        except IndexError:
            previous_good_index = y_good_indices[outlier_index]
        # Replace the outlier value with the previous good value
        X_test_arr[outlier_index] = X_test_arr[previous_good_index]

    # Concatenate the arrays into one NumPy array
    all_data_arr = np.concatenate((X_train_arr, X_test_arr), axis=0)

    # Convert the NumPy array back to a DataFrame
    all_data_df = pd.DataFrame(all_data_arr, columns=X_train.columns)

    return all_data_df

