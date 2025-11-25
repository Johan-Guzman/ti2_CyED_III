import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def save_plot(fig, filename, base_path="../outputs/lstm"):
    """
    Save a figure in the figures folder inside base_path.
    """
    figures_dir = os.path.join(base_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Figure saved at: {path}")

def save_trained_model(model, history, metrics_dict, figures=None, model_name="lstm_model", base_path="../outputs/lstm"):
    """
    Save model, weights, history, metrics and plots using a configurable base_path.
    
    Parameters:
    - model: trained model
    - history: training history
    - metrics_dict: dictionary with final metrics
    - figures: list of tuples [(fig_obj, filename), ...]
    - model_name: base name for saved files
    - base_path: root folder where everything will be saved
    """
    # Create subfolders if they don't exist
    saved_models_dir = os.path.join(base_path, "saved_models")
    metrics_dir = os.path.join(base_path, "metrics")
    figures_dir = os.path.join(base_path, "figures")

    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save full model
    model_full_path = os.path.join(saved_models_dir, f"{model_name}.h5")
    model.save(model_full_path)
    print(f"Full model saved at {model_full_path}")

    # Save only weights
    weights_path = os.path.join(saved_models_dir, f"{model_name}_weights.h5")
    model.save_weights(weights_path)
    print(f"Weights saved at {weights_path}")

    # Save training history
    history_path = os.path.join(saved_models_dir, f"{model_name}_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Training history saved at {history_path}")

    # Save metrics
    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved at {metrics_path}")

    # Save figures
    if figures is not None:
        for fig, filename in figures:
            save_plot(fig, filename, base_path=base_path)
