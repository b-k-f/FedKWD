import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from util import gradient_magnitude, weight_update_magnitude

# Defining loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error

def train_client(train_loader, global_model, server_logits, lr, batch_size, num_local_epochs, model_type, device):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

     # To track the cumulative loss
    aggregated_logits = []  # To store aggregated logits across batches
    # feature_maps = []  # To store feature maps

    for epoch in range(num_local_epochs):
        h = model.init_hidden(batch_size)
        # total_loss = 0.0  # Track loss for this epoch
        aggregated_loss = 0.0 
        for i, (x, label) in enumerate(train_loader):
            if model_type == "GRU":
                h = h.data
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])

            model.zero_grad()

            # Forward pass through the model
            out, h = model(x.to(device).float(), h)

            # Retrieve the corresponding batch of server logits
            server_logits_batch = server_logits[i].to(device).float()
            
            # Calculate local and distillation losses
            
            local_loss = criterion(out, label.to(device).float())
            distillation_loss = criterion(out, server_logits_batch)
            # distillation_loss=F.kl_div(F.log_softmax(out, dim=1).to(device), F.softmax(server_logits_batch).to(device),reduction='batchmean')          
            
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            # Total loss with L1, L2, and knowledge distillation
            locaL_loss = local_loss + distillation_loss + l1_lambda * l1_norm

            # total_loss = local_loss + distillation_loss
            # print("loss", total_loss)

            local_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            aggregated_logits.append(out.reshape(-1).cpu().detach().numpy())
            # feature_maps.append(h[0].cpu().detach().numpy())
            aggregated_loss += local_loss.item() * (1/len(train_loader))

        # Average loss for the epoch
        # epoch_loss /= len(train_loader.dataset)
        
        # aggregated_loss += epoch_loss
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, aggregated_loss))
    
    aggregated_logits = np.concatenate(aggregated_logits, axis=0)
    # feature_maps = np.concatenate(feature_maps, axis=1)
    # print("aggregated_logits ", np.array(aggregated_logits).shape)
    # print("feature_maps ", np.array(feature_maps).shape)

    return model, aggregated_loss, aggregated_logits

def evaluate(model, test_loader, label_sc, model_type, device):
    model.eval()
    model.to(device)
    all_out= []
    all_targ = []
    total_loss = 0.0
    
    for k, test_data in enumerate(test_loader):
        outputs = []
        targets = []
        for batch_x, batch_y in test_data:
            inputs = batch_x.to(device)
            labels = batch_y.to(device)
            if torch.isnan(inputs).any():
                print(f"NaN found in inputs at round {k}!")
            if torch.isnan(labels).any():
                print(f"NaN found in labels at round {k}!")
            # Move each tensor in the tuple to the same device as the model            
            # h = model.init_hidden(inputs.shape[0])
            if model_type == "LSTM":
                h = tuple(h_item.to(device) for h_item in model.init_hidden(inputs.shape[0]))
            if model_type == "GRU":
                h = model.init_hidden(inputs.shape[0]).to(device)

            with torch.no_grad():
                out, h = model(inputs.to(device).float(), h)
            if torch.isnan(out).any():
                print(f"NaN found in model output at round {k}!")
            outputs.append(out.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
            # Calculate the loss
            loss = criterion(out, labels)
            # total_loss += loss.item()
            total_loss += loss.item() * 1/len(test_data)
        if torch.isnan(out).any():
            print(f"NaN found in model output at {k}!")

        concatenated_outputs = np.concatenate(outputs)
        concatenated_targets = np.concatenate(targets)
        all_out.append(label_sc[k].inverse_transform(concatenated_outputs).reshape(-1))
        all_targ.append(label_sc[k].inverse_transform(concatenated_targets).reshape(-1))

    extra_conc_out= np.concatenate(all_out)
    extra_conc_targ= np.concatenate(all_targ)
    # print("Targets dtype:", type(extra_conc_targ), targets.dtype if isinstance(extra_conc_targ, np.ndarray) else "Not a NumPy array")
    # print("out dtype:", type(extra_conc_out), targets.dtype if isinstance(extra_conc_out, np.ndarray) else "Not a NumPy array")

    if np.isnan(np.array(extra_conc_targ, dtype=np.float32)).any():
        print("NaN values found in targets", k)

    if np.isnan(np.array(extra_conc_out,dtype=np.float32)).any():
        print("NaN values found in outputs", k)
    # Calculate and print other metrics
    smape = calculate_smape(extra_conc_out, extra_conc_targ)
    mae = mean_absolute_error(extra_conc_targ, extra_conc_out)
    rmse = np.sqrt(mean_squared_error(extra_conc_targ, extra_conc_out))
    print(f"calc smape: {smape}%")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Calculate and print the average loss
    average_loss = total_loss / len(test_loader)
    print(f"Average Loss: {average_loss: }")

    return all_out, all_targ, average_loss, smape, mae, rmse

def train_client_cadis(train_loader, global_model, lr, batch_size, num_local_epochs, model_type, lambda_kd,device):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    global_model.to(device)
    global_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_local_epochs):
        h_local = model.init_hidden(batch_size)
        total_loss = 0.0
        for x, label in train_loader:
            h_local = tuple(e.data for e in h_local)
            global_output, _ = global_model(x.to(device).float(), h_local)

            model.zero_grad()
            local_output, h_local = model(x.to(device).float(), h_local)
            mse_loss = criterion(local_output, label.to(device).float())

            # KL divergence as knowledge distillation loss
            kd_loss = F.kl_div(
                F.log_softmax(local_output, dim=1),
                F.softmax(global_output.detach(), dim=1),
                reduction='batchmean'
            )

            loss = mse_loss + lambda_kd * kd_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_local_epochs}], Loss: {total_loss:.4f}")
    return model

def train_client_gkd(train_loader, global_model, global_logits, lr, batch_size, num_local_epochs, model_type, device, kd_weight):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()  # For regression loss
    criterion_kd = nn.KLDivLoss(reduction="batchmean")  # Knowledge distillation loss
    
    for epoch in range(num_local_epochs):
        h = model.init_hidden(batch_size)
        for x, label in train_loader:
            if model_type == "GRU":
                h = h.data
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])
            model.zero_grad()
            out, h = model(x.to(device).float(), h)

            # Ensure label is FloatTensor (for regression tasks)
            label = label.to(device).float()

            # Compute regression loss (MSE)
            loss_mse = criterion_mse(out, label)

            # Knowledge distillation loss
            # loss_kd = criterion_kd(F.log_softmax(out, dim=1), F.softmax(global_logits, dim=1))
            loss_kd=F.kl_div(F.log_softmax(out, dim=1).to(device), F.softmax(global_logits).to(device),reduction='batchmean')          
            
            loss = loss_mse + kd_weight * loss_kd
            loss.backward()
            optimizer.step()
            
    return model

def train_first(train_loader, global_model, lr, batch_size, num_local_epochs, model_type, device):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    input_dim = next(iter(train_loader))[0].shape[2]  # 6
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    aggregated_logits=[]
    # Start training loop
    for epoch in range(0, num_local_epochs):
        h = model.init_hidden(batch_size)
        total_loss = 0.0
        for x, label in train_loader:
            if model_type == "GRU":
                h = h.data
            # Unpcak both h_0 and c_0
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])
            
            model.zero_grad()  # Set the gradients to zero before starting to do backpropragation
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())

            # Perform backpropragation
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (1/len(train_loader))
            aggregated_logits.append(out.reshape(-1).cpu().detach().numpy())

            # avg_loss += loss.item() * x.size(0)
        # avg_loss = avg_loss / len(train_loader.dataset)
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, total_loss))
    aggregated_logits = np.concatenate(aggregated_logits, axis=0)

    return model, total_loss, aggregated_logits

def train_csa(train_loader, global_model, lr, batch_size, num_local_epochs, model_type, device, previous_weights):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    input_dim = next(iter(train_loader))[0].shape[2]  # 6
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    avg_gradient_magnitude = 0.0
    avg_weight_update_magnitude = 0.0
    
    # Start training loop
    for epoch in range(0, num_local_epochs):
        h = model.init_hidden(batch_size)
        total_loss = 0.0
        for x, label in train_loader:
            if model_type == "GRU":
                h = h.data
            # Unpcak both h_0 and c_0
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])
            
            model.zero_grad()  # Set the gradients to zero before starting to do backpropragation
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())

            # Perform backpropragation
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (1/len(train_loader))
            # avg_loss += loss.item() * x.size(0)
            # print('x')
            # Compute gradients and weight updates
            gradient_magnitude_val = gradient_magnitude(model)
            weight_update_magnitude_val = weight_update_magnitude(model, previous_weights)
            avg_gradient_magnitude += gradient_magnitude_val * (1 / len(train_loader))
            avg_weight_update_magnitude += weight_update_magnitude_val * (1 / len(train_loader))

        # avg_loss = avg_loss / len(train_loader.dataset)
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, total_loss))
    # print(f'Avg Gradient Magnitude: {avg_gradient_magnitude:.4f},Avg Weight Update Magnitude: {avg_weight_update_magnitude:.4f}')

    return model, total_loss, avg_gradient_magnitude, avg_weight_update_magnitude

def train_csakd(train_loader, global_model, server_logits, lr, batch_size, num_local_epochs, model_type, device, previous_weights):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    input_dim = next(iter(train_loader))[0].shape[2]  # 6
     # To track the cumulative loss
    aggregated_logits = []  # To store aggregated logits across batches
    # feature_maps = []  # To store feature maps
    avg_gradient_magnitude = 0.0
    avg_weight_update_magnitude = 0.0
    
    for epoch in range(num_local_epochs):
        h = model.init_hidden(batch_size)
        # total_loss = 0.0  # Track loss for this epoch
        aggregated_loss = 0.0 
        for i, (x, label) in enumerate(train_loader):
            if model_type == "GRU":
                h = h.data
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])

            model.zero_grad()

            # Forward pass through the model
            out, h = model(x.to(device).float(), h)

            # Retrieve the corresponding batch of server logits
            server_logits_batch = server_logits[i].to(device).float()
            
            # Calculate local and distillation losses
            
            local_loss = criterion(out, label.to(device).float())
            distillation_loss = criterion(out, server_logits_batch)
            # distillation_loss=F.kl_div(F.log_softmax(out, dim=1).to(device), F.softmax(server_logits_batch).to(device),reduction='batchmean')          
            
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            # Total loss with L1, L2, and knowledge distillation
            locaL_loss = local_loss + distillation_loss + l1_lambda * l1_norm

            # total_loss = local_loss + distillation_loss
            # print("loss", total_loss)

            local_loss.backward()
            optimizer.step()

            aggregated_logits.append(out.reshape(-1).cpu().detach().numpy())
            # feature_maps.append(h[0].cpu().detach().numpy())
            aggregated_loss += local_loss.item() * (1/len(train_loader))
            
            gradient_magnitude_val = gradient_magnitude(model)
            weight_update_magnitude_val = weight_update_magnitude(model, previous_weights)
            avg_gradient_magnitude += gradient_magnitude_val * (1 / len(train_loader))
            avg_weight_update_magnitude += weight_update_magnitude_val * (1 / len(train_loader))
        # Average loss for the epoch
        # epoch_loss /= len(train_loader.dataset)
        
        # aggregated_loss += epoch_loss
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, aggregated_loss))
    
    aggregated_logits = np.concatenate(aggregated_logits, axis=0)
    # feature_maps = np.concatenate(feature_maps, axis=1)
    # print("aggregated_logits ", np.array(aggregated_logits).shape)
    # print("feature_maps ", np.array(feature_maps).shape)

    return model, aggregated_loss, aggregated_logits, avg_gradient_magnitude, avg_weight_update_magnitude


def evaluate_csa(model, test_loader, label_sc, model_type, device, previous_weights):
    model.eval()
    model.to(device)
    all_out= []
    all_targ = []
    total_loss = 0.0
    avg_weight_update_magnitude = 0.0
    for k, test_data in enumerate(test_loader):
        outputs = []
        targets = []
        for batch_x, batch_y in test_data:
            inputs = batch_x.to(device)
            labels = batch_y.to(device)
            # Move each tensor in the tuple to the same device as the model            
            # h = model.init_hidden(inputs.shape[0])
            if model_type == "LSTM":
                h = tuple(h_item.to(device) for h_item in model.init_hidden(inputs.shape[0]))
            if model_type == "GRU":
                h = model.init_hidden(inputs.shape[0]).to(device)

            with torch.no_grad():
                out, h = model(inputs.to(device).float(), h)

            outputs.append(out.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
            # Calculate the loss
            loss = criterion(out, labels)
            # total_loss += loss.item()
            total_loss += loss.item() * 1/len(test_data)
            
            # Compute gradients and weight updates
            weight_update_magnitude_val = weight_update_magnitude(model, previous_weights)
            avg_weight_update_magnitude += weight_update_magnitude_val * (1 / len(test_data))

        concatenated_outputs = np.concatenate(outputs)
        concatenated_targets = np.concatenate(targets)
        all_out.append(label_sc[k].inverse_transform(concatenated_outputs).reshape(-1))
        all_targ.append(label_sc[k].inverse_transform(concatenated_targets).reshape(-1))

    extra_conc_out= np.concatenate(all_out)
    extra_conc_targ= np.concatenate(all_targ)
    # Calculate and print other metrics
    smape = calculate_smape(extra_conc_out, extra_conc_targ)
    mae = mean_absolute_error(extra_conc_targ, extra_conc_out)
    rmse = np.sqrt(mean_squared_error(extra_conc_targ, extra_conc_out))
    print(f"calc smape: {smape}%")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Calculate and print the average loss
    average_loss = total_loss / len(test_loader)
    print(f"Average Loss: {average_loss: }")
    # print(f'test Weight Update Magnitude: {avg_weight_update_magnitude:.4f}')
    return all_out, all_targ, average_loss, smape, mae, rmse, avg_weight_update_magnitude
def running_model_sum(current, next):
    if current == None:
        current = next
    else:
        for key in current:
            current[key] = current[key] + next[key]
    return current

def scale_model_state(model_state, scale):
    scaled_state = {key: value * scale for key, value in model_state.items()}
    return scaled_state
    
def calculate_smape(forecasted, actual):
    # Check for equal length of forecasted and actual arrays
    if len(forecasted) != len(actual):
        raise ValueError("Forecasted and actual arrays must have the same length.")
    total_smape = 0.0
    
    for i in range(len(forecasted)):
        Ft = forecasted[i]
        At = actual[i]
        numerator = abs(Ft - At)
        denominator = abs(Ft) + abs(At)
        
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        smape_i = (numerator / (denominator + epsilon)) * 100.0
        total_smape += smape_i
    
    # Calculate the average SMAPE over all data points
    average_smape = total_smape / len(forecasted)
    
    return average_smape
