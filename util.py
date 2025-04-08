import random
import copy
import numpy as np
import pandas as pd
np.random.seed(0)
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import torch.nn.init as init
from scipy.cluster.hierarchy import linkage, fcluster
import torch

def pad_logits(logits_list):
    max_length = max(logits.shape[0] for logits in logits_list)
    padded_logits = []

    for logits in logits_list:
        padded = np.pad(logits, (0, max_length - logits.shape[0]), mode='constant', constant_values=1e-8)  # Small constant instead of 0
        
        # Ensure values are non-negative
        padded = np.maximum(padded, 1e-8)

        total_sum = np.sum(padded)
        if total_sum > 1e-8:
            padded /= total_sum  # Normalize to sum to 1
        else:
            padded[:] = 1.0 / len(padded)   # Assign uniform distribution if sum is zero
        
        padded_logits.append(padded)

    return np.array(padded_logits)

def compute_jsd_matrix(logits):
    n = len(logits)
    D = np.zeros((n, n))

    for i in range(n):
        # Ensure logits are valid probability distributions
        if np.any(logits[i] < 0) or np.sum(logits[i]) == 0:
            print(f"Invalid logits at index {i}: {logits[i]}")
            logits[i] = np.full_like(logits[i], 1.0 / len(logits[i]))  # Assign uniform distribution

        for j in range(n):
            if i != j:
                D[i, j] = jensenshannon(logits[i], logits[j])

    return D

def determine_optimal_clusters(jsd_matrix, max_clusters=10):
    # Compute similarity matrix from JSD distances
    similarity_matrix = np.exp(-jsd_matrix**2)
    
    # Compute eigenvalues of the normalized Laplacian
    L = laplacian(similarity_matrix, normed=True)
    eigenvalues, _ = eigh(L)

    # Calculate eigenvalue gaps
    gaps = np.diff(eigenvalues[:max_clusters])  # Focus on the smallest eigenvalues
    optimal_k = np.argmax(gaps) + 1  # Index of largest gap + 1

    return optimal_k

def perform_spectral_clustering(jsd_matrix, num_clusters):
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = clustering.fit_predict(np.exp(-jsd_matrix**2)+1e-8)  # Gaussian similarity
    return labels

def compute_cluster_irregularities(jsd_matrix, labels):
    cluster_irregularities = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        distances = jsd_matrix[np.ix_(cluster_indices, cluster_indices)]
        intra_cluster_mean = np.mean(distances)
        cluster_irregularities[label] = intra_cluster_mean

    # Safe normalization
    min_irreg = min(cluster_irregularities.values())
    max_irreg = max(cluster_irregularities.values())
    if max_irreg > min_irreg:
        for label in cluster_irregularities:
            cluster_irregularities[label] = (cluster_irregularities[label] - min_irreg) / (max_irreg - min_irreg + 1e-8)
    else:
        for label in cluster_irregularities:
            cluster_irregularities[label] = 0.0  # Set to zero if range is zero

    return cluster_irregularities

def assign_weights(labels, cluster_irregularities, data_sizes):
    weights = np.zeros(len(labels))
    for i, label in enumerate(labels):
        irregularity = cluster_irregularities[label]
        W_c = 1 / (irregularity + 1e-8)  # Compute cluster weight
        weights[i] = data_sizes[i] * W_c
    
    # Normalize weights
    weights /= np.sum(weights)
    return weights

def move_sliding_window(data, window_size, inputs_cols_indices, label_col_index):
    """
    data: numpy array including data
    window_size: size of window
    inputs_cols_indices: col indices to include
    """

    # (# instances created by movement, seq_len (timestamps), # features (input_len))
    inputs = np.zeros((len(data) - window_size, window_size, len(inputs_cols_indices)))
    labels = np.zeros(len(data) - window_size)

    for i in range(window_size, len(data)):
        inputs[i - window_size] = data[i - window_size : i, inputs_cols_indices]
        labels[i - window_size] = data[i, label_col_index]
    inputs = inputs.reshape(-1, window_size, len(inputs_cols_indices))
    labels = labels.reshape(-1, 1)
    print(inputs.shape, labels.shape)

    return inputs, labels

def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#############cadis###########################
def calculate_similarity_matrix(global_model, client_models,device):
    """
    Calculate the Q-matrix (similarity matrix) using penultimate layers of client models.
    """
    q_matrix = np.zeros((len(client_models), len(client_models)))
    penultimate_layers = []

    global_model.to(device)
    for model in client_models:
        model.to(device)
        penultimate_layer = list(model.state_dict().values())[-2]  # Penultimate layer assumed to be second last
        penultimate_layers.append(penultimate_layer.cpu().numpy())

    for i, client_i in enumerate(penultimate_layers):
        for j, client_j in enumerate(penultimate_layers):
            if i != j:
                similarity = np.dot(client_i.flatten(), client_j.flatten()) / (
                    np.linalg.norm(client_i.flatten()) * np.linalg.norm(client_j.flatten()) + 1e-8
                )
                q_matrix[i, j] = similarity

    return q_matrix
def cluster_clients(q_matrix, threshold):
    """
    Perform clustering based on the Q-matrix similarity values.
    """
    linkage_matrix = linkage(q_matrix, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    return clusters
def clustered_aggregation(global_model, client_models, q_matrix, client_data_sizes, threshold, device):
    """
    Aggregate client models based on their cluster membership and sizes.
    """
    clusters = cluster_clients(q_matrix, threshold=threshold)  # Use the provided threshold
    cluster_sizes = {c: sum(1 for x in clusters if x == c) for c in set(clusters)}
    
    aggregated_weights = None
    for i, model in enumerate(client_models):
        weight_factor = 1 / cluster_sizes[clusters[i]] * (client_data_sizes[i] / sum(client_data_sizes))
        state_dict = model.state_dict()
        
        # Initialize aggregated_weights if not already done
        if aggregated_weights is None:
            aggregated_weights = {key: torch.zeros_like(value) for key, value in state_dict.items()}
        
        # Accumulate weighted contributions
        for key in state_dict:
            aggregated_weights[key] += state_dict[key] * weight_factor

    # Load the aggregated weights into the global model
    global_model.load_state_dict(aggregated_weights)
    global_model.to(device)
    return global_model

##############GKD###########################
def average_model_weights(models):
    avg_weights = copy.deepcopy(models[0])
    for key in avg_weights:
        for model in models[1:]:
            avg_weights[key] += model[key]
        avg_weights[key] = avg_weights[key] / len(models)
    return avg_weights

# def extract_logits(model, data_loader, device):
#     model.eval()
#     logits = []
#     with torch.no_grad():
#         for x, _ in data_loader:
#             x = x.to(device)
#             out, _ = model(x)
#             logits.append(out.cpu())
#     return torch.cat(logits, dim=0)

def extract_logits(model, data_loader, device):
    model.eval()
    model.to(device)
    
    logits = []
    with torch.no_grad():
        for batch in data_loader:
            for x, label in batch:
                # Ensure input tensor is FloatTensor and move it to the correct device
                x = x.to(device).float()

                # Initialize hidden state on the correct device
                h = tuple(h_item.to(device) for h_item in model.init_hidden(x.size(0)))

                # Forward pass with input and hidden state
                out, _ = model(x, h)

                # Append reshaped logits to the list
                logits.append(out.cpu().reshape(-1))

    return torch.cat(logits, dim=0)

########fedlbs########
def swap_losses(lst, wgt):
    # Create lists of tuples (index, loss) and (index, weight)
    indexed_lst = list(enumerate(lst))
    indexed_wgt = list(enumerate(wgt))

    # Sort the list of tuples by loss values
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1])
    # Create a mapping of old index to new index
    index_mapping = {sorted_lst[i][0]: sorted_lst[-(i + 1)][0] for i in range(len(sorted_lst) // 2)}
    # Swap elements
    for old_index, new_index in index_mapping.items():
        lst[old_index], lst[new_index] = lst[new_index], lst[old_index]
        wgt[old_index], wgt[new_index] = wgt[new_index], wgt[old_index]
    return wgt

########fedcsa###########
def client_selection(values, threshold):
    # Step 1: Identify the maximum value
    max_value = max(values)
    # Step 2: Define the threshold value
    threshold_value = max_value * threshold
    # Step 3: Collect indices of values that are within the threshold of the maximum value
    indices = [index for index, value in enumerate(values) if value >= threshold_value]
    return indices

def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def objective_f(loss, grad_magnitude, weight_upd_magnitude, autoenc):
    norm_gradient_magnitudes = normalize(np.array(grad_magnitude))
    norm_weight_update_magnitudes = normalize(np.array(weight_upd_magnitude))
    norm_loss = normalize(np.array(loss))
    norm_ae = normalize(np.array(autoenc))
    
    # Invert the normalized values of the magnitudes to make them minimizing objectives
    inv_norm_gradient_magnitudes = 1 - norm_gradient_magnitudes
    inv_norm_weight_update_magnitudes = 1 - norm_weight_update_magnitudes
    inv_norm_loss = 1 - norm_loss
    inv_norm_ea = 1 - norm_ae
    # Combine metrics into a single score for each client without accuracy
    alpha, gamma, delta = 0.34,0.33,0.33  # Ensure weights sum to 1
    raw_scores = (
        alpha * inv_norm_gradient_magnitudes +
        # beta * inv_norm_weight_update_magnitudes +
        gamma * inv_norm_loss + 
        delta * inv_norm_ea
    )
    # Normalize the raw scores to sum to 1
    total_score = np.sum(raw_scores)
    proportionate_scores = raw_scores / total_score

    return proportionate_scores

def gradient_magnitude(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def weight_update_magnitude(model, previous_weights):
    update_magnitude = 0.0
    for p, prev_p in zip(model.parameters(), previous_weights):
        update_magnitude += torch.sum((p.data - prev_p.data) ** 2).item()
    return update_magnitude

def save_weights(model):
    return [p.data.clone() for p in model.parameters()]
