import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory
base_path = r"\SLEEP\results_ensemble_multibranch"
branches = ['mfcc', 'mel', 'cqt']

for branch in branches:
    # File paths
    prob_path = os.path.join(base_path, f"y_prob_{branch}.npy")
    true_path = os.path.join(base_path, "y_true.npy")

    # Load predictions and true labels
    y_prob = np.load(prob_path)
    y_true = np.load(true_path)
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)

    # Separate confidence scores
    correct_conf = y_conf[y_pred == y_true]
    wrong_conf = y_conf[y_pred != y_true]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(correct_conf, color='green', label='Correct Predictions', kde=True, stat='density', bins=20)
    sns.histplot(wrong_conf, color='red', label='Incorrect Predictions', kde=True, stat='density', bins=20)
    plt.title(f'{branch.upper()} Branch - Prediction Confidence Distribution')
    plt.xlabel('Confidence Score (Softmax)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Save as PNG
    save_path = os.path.join(base_path, f"confidence_hist_{branch}.png")
    plt.savefig(save_path)
    plt.close()
