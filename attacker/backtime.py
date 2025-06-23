import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.append('ics-anomaly-detection-main')  # Add folder to system path

import data_loader, utils
import torch 
import numpy as np 
import pickle
import matplotlib.pyplot as plt

#BACKTIME requirements
import f_s
import f_g

# BACKTIME Algorithm:
# INPUT: Dataset X, forecasting model f_s, trigger generator f_g, injection rate alpha_T, target variable set S

def attack(dataset_name, backdoor_features):
    # BACKTIME Algorithm:
    # INPUT: Dataset X, forecasting model f_s, trigger generator f_g, injection rate alpha_T, target variable set S

    # Load dataset
    X_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    X_val, _, _, _ = utils.custom_train_test_split(dataset_name, x_test, y_test)

    # Train warm-up surrogate model
    ae = f_s.BackTimeAE(nI=X_train.shape[1])
    ae.train(X_train, X_val)

    # Get top alpha_T error timestamps
    alpha_T = 50
    T = X_train
    D = T.shape[1]
    top_ts_indices, _ = ae.select_top_alpha_timestamps(X_train, alpha_T)
    print("Top MAE indices:", top_ts_indices)

    # Construct adjacency matrix and initialize trigger generator
    A = f_g.compute_adjacency_matrix(X_train)
    trigger_gen = f_g.TriggerGeneratorGCN(input_dim=D, hidden_dim=64, output_dim=D, adjacency_matrix=A)
    optimizer = torch.optim.Adam(trigger_gen.parameters(), lr=0.001)

    # Bi-level optimization
    epoch_train = 20
    for epoch in range(epoch_train):
        print(f"\n[Epoch {epoch + 1}/{epoch_train}] Training...")

        # Generate Trigger
        identity_input = torch.eye(D)  # Identity matrix as node features
        trigger_output = trigger_gen(identity_input)
        trigger_vector = trigger_output.detach().numpy().diagonal()  # Use diagonal as trigger

        # Inject Trigger into selected features at top timestamps
        X_poisoned = X_train.copy()
        for t in top_ts_indices:
            if t < len(X_poisoned):
                X_poisoned[t, backdoor_features] += trigger_vector[backdoor_features]

        # Update Surrogate Model f_s
        surrogate_model = f_s.BackTimeAE(nI=X_train.shape[1], history_length=10)
        surrogate_model.train(X_poisoned, X_val)

        # Update Trigger Generator f_g via surrogate loss
        X_seq = surrogate_model.prepare_data(X_poisoned)
        predictions = surrogate_model.model.predict(X_seq)
        target_seq = np.zeros_like(predictions)  # Attacker-defined target (e.g., zeros)

        loss_val = np.mean((predictions - target_seq) ** 2)
        loss_tensor = torch.tensor(loss_val, requires_grad=True)

        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()

        print(f"[Trigger Loss]: {loss_val:.6f}")

    # Final trigger application (optional)
    final_trigger_output = trigger_gen(identity_input)
    final_trigger_vector = final_trigger_output.detach().numpy().diagonal()
    X_ATK = X_train.copy()
    for t in top_ts_indices:
        if t < len(X_ATK):
            X_ATK[t, backdoor_features] += final_trigger_vector[backdoor_features]

    print("[BackTime] Attack Data Generated.")
    
    # Save X_ATK
    atk_path = f"X_ATK_{dataset_name}.pkl"
    with open(atk_path, 'wb') as f:
        pickle.dump(X_ATK, f)
    print(f"[BackTime] X_ATK saved to {atk_path}")
    
def plot_full_timeseries_clean(X_orig, X_poisoned, features, save_path="full_timeseries_clean.png", max_len=5000):
    num_feats = len(features)
    fig, axs = plt.subplots(num_feats, 1, figsize=(15, 3 * num_feats), sharex=True)

    if num_feats == 1:
        axs = [axs]  # Ensure it's iterable

    for i, feat in enumerate(features):
        axs[i].plot(X_orig[:max_len, feat], label=f'Original F{feat}', linestyle='--', linewidth=1)
        axs[i].plot(X_poisoned[:max_len, feat], label=f'Backdoored F{feat}', linestyle='-', linewidth=1)
        axs[i].set_ylabel(f'Feature {feat}')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestamps")
    plt.suptitle("BackTime - Full Time Series Comparison (Simplified)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Simplified Timeseries Plot Saved] {save_path}")

