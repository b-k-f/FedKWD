import numpy as np
import zlib
import pickle
import torch 
import torch.nn as nn
import copy
from scipy.cluster.hierarchy import linkage, fcluster

from train import train_client,train_first, running_model_sum, scale_model_state, evaluate, train_client_cadis, train_client_gkd, train_csa, evaluate_csa, train_csakd
from util import compute_jsd_matrix, determine_optimal_clusters, perform_spectral_clustering, compute_cluster_irregularities, assign_weights, pad_logits, calculate_similarity_matrix, cluster_clients, clustered_aggregation, average_model_weights, extract_logits, swap_losses, objective_f, client_selection, save_weights
from ad_method import Autoencoder, detect_anomalies_with_autoencoder, flatten_weights

def fedcsakd(global_model, aggregated_logits, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    weight_lst = []

    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    stability_threshold = 10
    tmp_cl = clients
    n_tmp_cl = len(tmp_cl)
    tmp_cl_cid =  [clients[i] for i in tmp_cl]
    copy_test_loader = test_loader
    absence_count = {i: 0 for i in range(n_clients)}
    selection_threshold = 0.5
    for t in range(1, max_rounds):
        weight_avg = None
        client_logits = []
        # client_feature_maps = []
        data_sizes = []  # Track client data sizes
        client_models = []
        
        cl_grad_mag = []
        cl_upd_mag = []
        loss_lst = []
        ae_lst = []

        print("\nstarting round {}".format(t))
        print("clients: ", clients)
        previous_global_weights = save_weights(global_model)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu
        
        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))
            if t ==1:
                previous_weights = previous_global_weights
            else:
                previous_weights = save_weights(weight_lst[k])            
            
            # Each client uses the aggregated logits from the previous round (if available)
            local_model, train_loss,logits, train_gradient_magnitude, train_weight_update_magnitude = train_csakd(
                train_loader=client_train_loader[k],
                global_model=global_model,
                server_logits=aggregated_logits,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device,
                previous_weights=previous_weights
            )
            client_logits.append(logits)  # Aggregate logits from batches
            client_models.append(local_model)
            # client_logits=torch.tensor(client_logits)
            # client_feature_maps.append(feature_maps)
            data_sizes.append(len(client_train_loader[k].dataset))
            # weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            cl_upd_mag.append(train_gradient_magnitude)
            cl_grad_mag.append(train_weight_update_magnitude)
            loss_lst.append(train_loss)
            weight_lst.append(local_model)
            
            is_anomaly, reconstruction_error = detect_anomalies_with_autoencoder(global_model, threshold=0.005)
            print(f"loss error: {reconstruction_error:.4f}")
            ae_lst.append(reconstruction_error)

        
        padded_client_logits = pad_logits(client_logits)

        jsd_matrix = compute_jsd_matrix(padded_client_logits)
        num_clusters = determine_optimal_clusters(jsd_matrix)
        labels = perform_spectral_clustering(jsd_matrix, num_clusters)
        # print(num_clusters, labels)
        
        # Compute cluster irregularities and assign weights
        cluster_irregularities = compute_cluster_irregularities(jsd_matrix, labels)
        weights = assign_weights(labels, cluster_irregularities, data_sizes)

        # Aggregate logits across clients
        aggregated_logits = torch.tensor(np.average(padded_client_logits, axis=0, weights=weights), dtype=torch.float32, device=device)

        # Scale the model weights using the computed weights
        for k,i in enumerate(clients):
            scaled_state = scale_model_state(client_models[k].state_dict(), weights[k])
            # if i in tmp_cl_cid:      
            #     weight_avg = running_model_sum(weight_avg, scaled_state)
            weight_avg = running_model_sum(weight_avg, scaled_state)
            
            
        global_model.load_state_dict(weight_avg)

        # print(aggregated_logits.shape)
        # Evaluate the global model
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)

        # Save metrics for this round
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

        scores = objective_f(loss_lst, cl_grad_mag, cl_upd_mag, ae_lst)
        # print(scores)
        
        tmp_cl = client_selection(scores, selection_threshold)
        n_tmp_cl = len(tmp_cl)
        print(f"tmp_cl { tmp_cl}, n_tmp_cl {n_tmp_cl}")
        tmp_cl_cid =  [clients[i] for i in tmp_cl]
        print("tmp_cl_cid", tmp_cl_cid)
        # Update absence count for each client
        current_set = set(tmp_cl)
        for i, client in enumerate(clients):
            # print(i, client)
            if i not in current_set:
                absence_count[client] += 1
            else:
                absence_count[client] = 0
            if absence_count[client] >= stability_threshold:
                del absence_count[client]
                print('deleted client ' , client)
        # Remove clients whose absence count exceeds the threshold
        clients = [client for client in clients if client in absence_count]
        n_clients = len(clients)
        print(absence_count)
        print("Updated clients: ", clients)

        test_loader = [copy_test_loader[i] for i in tmp_cl] 
        
    # Return aggregated outputs, targets, and metrics
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedlbskd(global_model, aggregated_logits, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device, S):
    all_smape = []
    all_loss = []
    all_mae = []
    all_rmse = []
    loss_lst = [0.0] * n_clients
    weight_lst = [0.0] * n_clients
    loss = 0.0
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    
    for t in range(1, max_rounds):
        weight_avg = None
        client_logits = []
        data_sizes = []  # Track client data sizes
        client_models = []
        
        rnd= "average"

        if t % S == 0 and t!=0:
            # swap= swap_weights(dict_idl)
            swap = swap_losses(loss_lst, weight_lst)
            rnd= "swapping"
        
        print("\nstarting {} round {}".format(rnd, t))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu
        
        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))

            if rnd == "swapping":
                global_model.load_state_dict(pickle.loads(zlib.decompress(swap[k])))
            
            # Each client uses the aggregated logits from the previous round (if available)
            local_model, train_loss,logits = train_client(
                train_loader=client_train_loader[k],
                global_model=global_model,
                server_logits=aggregated_logits,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device
            )
            client_logits.append(logits)  # Aggregate logits from batches
            client_models.append(local_model)
            # client_logits=torch.tensor(client_logits)
            # client_feature_maps.append(feature_maps)
            data_sizes.append(len(client_train_loader[k].dataset))
            # weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            
            if (t+1) % S == 0:
                print('updating loss and weight')
                loss_lst[k] = train_loss
                weight_lst[k] = zlib.compress(pickle.dumps(local_model.state_dict()))

        padded_client_logits = pad_logits(client_logits)

        jsd_matrix = compute_jsd_matrix(padded_client_logits)
        num_clusters = determine_optimal_clusters(jsd_matrix)
        labels = perform_spectral_clustering(jsd_matrix, num_clusters)
        print(num_clusters, labels)
        
        # Compute cluster irregularities and assign weights
        cluster_irregularities = compute_cluster_irregularities(jsd_matrix, labels)
        weights = assign_weights(labels, cluster_irregularities, data_sizes)

        # Aggregate logits across clients
        aggregated_logits = torch.tensor(np.average(padded_client_logits, axis=0, weights=weights), dtype=torch.float32, device=device)
        # Scale the model weights using the computed weights
        for i in range(n_clients):
            scaled_state = scale_model_state(client_models[i].state_dict(), weights[i])
            # scaled_state = scale_model_state(client_models[i].state_dict(), weights[i])
            weight_avg = running_model_sum(weight_avg, scaled_state)

        global_model.load_state_dict(weight_avg)

        # print(aggregated_logits.shape)
        # Evaluate the global model
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)

        # Save metrics for this round
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Return aggregated outputs, targets, and metrics
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedkd(global_model, aggregated_logits, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    all_smape = []
    all_loss = []
    all_mae = []
    all_rmse = []
    
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    
    for t in range(1, max_rounds):
        weight_avg = None
        client_logits = []
        # client_feature_maps = []
        data_sizes = []  # Track client data sizes
        client_models = []

        print("\nstarting round {}".format(t))
        print("clients: ", clients)
        
        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))
            
            # Each client uses the aggregated logits from the previous round (if available)
            local_model, _,logits = train_client(
                train_loader=client_train_loader[k],
                global_model=global_model,
                server_logits=aggregated_logits,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device
            )
            client_logits.append(logits)  # Aggregate logits from batches
            client_models.append(local_model)
            # client_logits=torch.tensor(client_logits)
            # client_feature_maps.append(feature_maps)
            data_sizes.append(len(client_train_loader[k].dataset))
            # weight_avg = running_model_sum(weight_avg, local_model.state_dict())

        padded_client_logits = pad_logits(client_logits)

        jsd_matrix = compute_jsd_matrix(padded_client_logits)
        num_clusters = determine_optimal_clusters(jsd_matrix)
        labels = perform_spectral_clustering(jsd_matrix, num_clusters)
        print(num_clusters, labels)
        
        # Compute cluster irregularities and assign weights
        cluster_irregularities = compute_cluster_irregularities(jsd_matrix, labels)
        weights = assign_weights(labels, cluster_irregularities, data_sizes)

        # Aggregate logits across clients
        aggregated_logits = torch.tensor(np.average(padded_client_logits, axis=0, weights=weights), dtype=torch.float32, device=device)
        # Scale the model weights using the computed weights
        for i in range(n_clients):
            scaled_state = scale_model_state(client_models[i].state_dict(), weights[i])
            # scaled_state = scale_model_state(client_models[i].state_dict(), weights[i])
            weight_avg = running_model_sum(weight_avg, scaled_state)

        global_model.load_state_dict(weight_avg)

        # print(aggregated_logits.shape)
        # Evaluate the global model
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)

        # Save metrics for this round
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Return aggregated outputs, targets, and metrics
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def cadis(global_model, client_train_loader, test_loader,lambda_kd, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    all_smape = []
    all_loss = []
    all_mae = []
    all_rmse = []
    
    client_data_sizes = [len(loader.dataset) for loader in client_train_loader]
    lambda_kd = 1
    threshold = 0.9
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    
    for t in range(max_rounds):
        print("\nstarting round {}".format(t))
        print("clients: ", clients)
        
        # Train client models with KD
        client_models = []
        for k,cid in enumerate(clients):

            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))
            
            client_model = train_client_cadis(
                train_loader=client_train_loader[k],
                global_model=global_model,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type="LSTM",  # Example model type
                lambda_kd=lambda_kd,
                device=device
            )
            client_models.append(client_model)

        # Calculate Q-matrix
        q_matrix = calculate_similarity_matrix(global_model, client_models,device)

        # Clustered aggregation
        global_model = clustered_aggregation(global_model, client_models, q_matrix, client_data_sizes, threshold,device)
        # Evaluate the global model
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)

        # Save metrics for this round
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Return aggregated outputs, targets, and metrics
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedgkd(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device, buffer_size, kd_weight):
    all_smape, all_loss, all_mae, all_rmse = [], [], [], []
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    model_buffer = []

    for t in range(max_rounds):
        weight_avg = None
        print(f"\nStarting round {t}")
        print("clients: ", clients)
        
        # Ensemble global models from buffer
        if len(model_buffer) > 0:
            ensemble_model = average_model_weights(model_buffer)
        else:
            ensemble_model = global_model.state_dict()

        global_logits = extract_logits(global_model, client_train_loader, device)

        # Update local clients
        for k, cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))
            local_model,_,_ = train_client(
                train_loader=client_train_loader[k],
                global_model=global_model,
                server_logits=global_logits,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device,
            )
            weight_avg = running_model_sum(weight_avg, local_model.state_dict())
        
        # Scale weights and update global model
        weight_avg = scale_model_state(weight_avg, 1 / n_clients)
        global_model.load_state_dict(weight_avg)
        
        # Update model buffer
        if len(model_buffer) >= buffer_size:
            model_buffer.pop(0)
        model_buffer.append(copy.deepcopy(global_model.state_dict()))
        
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedavg(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    client_logits = []
    
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    for t in range(max_rounds):

        weight_avg = None
        print("\nstarting avg round {}".format(t))
        # clients = list(range(0, n_clients))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,n_clients, cid))
            
            local_model, _, logits = train_first(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device)
            client_logits.append(logits)  # Aggregate logits from batches

            # do the average of local model weights with each client in a single round
            # weight_avg = running_model_avg(weight_avg, local_model.state_dict(), 1/n_clients)
            weight_avg = running_model_sum(weight_avg, local_model.state_dict())

        weight_avg = scale_model_state(weight_avg, 1/n_clients)
   
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)       
        padded_client_logits = pad_logits(client_logits)

        aggregated_logits = torch.tensor(np.average(padded_client_logits, axis=0), dtype=torch.float32, device=device)
        
        #save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse, global_model,aggregated_logits

def fedcsa(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    weight_lst = []

    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    stability_threshold = 30
    tmp_cl = clients
    n_tmp_cl = len(tmp_cl)
    copy_test_loader = test_loader
    absence_count = {i: 0 for i in range(n_clients)}

    for t in range(max_rounds):    
        cl_grad_mag = []
        cl_upd_mag = []
        loss_lst = []
        ae_lst = []
    
        weight_avg = None
        print("\nstarting round {}".format(t))
        # clients = list(range(0, n_clients))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu
        previous_global_weights = save_weights(global_model)

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k,n_clients-1, cid))
            if t ==0:
                previous_weights = previous_global_weights
            else:
                previous_weights = save_weights(weight_lst[k])
            
            local_model, train_loss, train_gradient_magnitude, train_weight_update_magnitude = train_csa(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device, 
                previous_weights = previous_weights)
            
            cl_upd_mag.append(train_gradient_magnitude)
            cl_grad_mag.append(train_weight_update_magnitude)
            loss_lst.append(train_loss)
            weight_lst.append(local_model)
            
            is_anomaly, reconstruction_error = detect_anomalies_with_autoencoder(global_model, threshold=0.005)
            print(f"loss error: {reconstruction_error:.4f}")
            ae_lst.append(reconstruction_error)
            
            # if k in tmp_cl:
            #     weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            
        weight_avg = scale_model_state(weight_avg, 1/(n_clients))
        # weight_avg = scale_model_state(weight_avg, 1/(n_tmp_cl))
        
        # for k,i in enumerate(clients):
        #     scaled_state = scale_model_state(local_model.state_dict(),1/(n_clients) )
        #     # if i in tmp_cl_cid:      
        #     #     weight_avg = running_model_sum(weight_avg, scaled_state)
        #     weight_avg = running_model_sum(weight_avg, scaled_state)
        
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse, test_weight_update_magnitude = evaluate_csa(global_model, test_loader, label_sc, model_type, device, previous_global_weights)       
        
        # print('global gradient magnitude: ', gradient_magnitude(global_model))
        # print('global weight update magnitude: ', gradient_magnitude(global_model))
        
        #save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

        scores = objective_f(loss_lst, cl_grad_mag, cl_upd_mag, ae_lst)
        
        tmp_cl = client_selection(scores, 0.75)
        n_tmp_cl = len(tmp_cl)

        # Update absence count for each client
        current_set = set(tmp_cl)
        for i, client in enumerate(clients):
            if i not in current_set:
                absence_count[client] += 1
            else:
                absence_count[client] = 0
            if absence_count[client] >= stability_threshold:
                del absence_count[client]
                print('deleted client ' , client)
        # Remove clients whose absence count exceeds the threshold
        clients = [client for client in clients if client in absence_count]
        n_clients = len(clients)
        print(absence_count)
        print("Updated clients: ", clients)

        test_loader = [copy_test_loader[i] for i in tmp_cl] 
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedlbs(global_model, client_train_loader, test_loader, label_sc, n_clients, num_clients_per_round, batch_size, num_local_epochs, lr, max_rounds, model_type, device, S):
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    loss_lst = [0.0] * num_clients_per_round
    weight_lst = [0.0] * num_clients_per_round
    loss = 0.0
    clients = np.arange(num_clients_per_round)
    for t in range(max_rounds):
        # ratio = num_clients_per_round
        rnd= "average"
        weight_avg = None
        if t % S == 0 and t!=0:
            # swap= swap_weights(dict_idl)
            swap = swap_losses(loss_lst, weight_lst)
            rnd= "swapping"
        
        print("\nstarting {} round {}".format(rnd, t))
        # clients = list(range(0, num_clients_per_round))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,num_clients_per_round, cid))
            
            if rnd == "swapping":
                global_model.load_state_dict(pickle.loads(zlib.decompress(swap[k])))
                
            local_model, train_loss,_ = train_first(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device)

            weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            
            # if the next round is swapping
            if (t+1) % S == 0:
                print('updating loss and weight')
                loss_lst[k] = train_loss
                weight_lst[k] = zlib.compress(pickle.dumps(local_model.state_dict()))
        
        weight_avg = scale_model_state(weight_avg, 1/num_clients_per_round)
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse  = evaluate(global_model, test_loader, label_sc, model_type,device)       
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse
