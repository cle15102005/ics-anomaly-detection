from sklearn import metrics
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score

def get_confusion_matrix(y_test, y_pred):
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"]
    )
    return matrix

def show_classification_report(y_test, y_pred):
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    
def show_confusion_matrix(matrix):
    print("(+) Getting confusion matrix...")
    cell_colors = [
        ["lightgreen", "lightcoral"],   # Row for actual normal (TN, FP)
        ["lightcoral", "lightgreen"]      # Row for actual anomaly (FN, TP)
    ]

    _, ax = plt.subplots(figsize=(6, 4))

    # Draw boxes manually with colors
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            color = cell_colors[i][j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            ax.text(j + 0.5, i + 0.5, str(value), va='center', ha='center',
                    fontsize=14, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted Normal", " Predicted Anomaly"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["True Normal", "True Anomaly"])
    #ax.set_xlabel("Predicted Label")
    #ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", pad=20)
    
    # Reverse y-axis and remove frame ticks
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.show()
    
def plot_evaluation(y_test, y_pred):
    print("(+) Ploting...")
    f1= f1_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    
    metrics_text = f'F1_Score= {f1:.2f}, prec= {precision:.2f}, rec= {recall:.2f}'
    
    _, ax = plt.subplots(figsize=(20, 4))
    ax.set_title(f'Comparing y_pred and y_test ({metrics_text})', fontsize = 25, pad = 25)
    ax.plot(-1 * y_pred, color = '0.25', label = 'Predicted')
    ax.plot(y_test, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
    ax.fill_between(np.arange(len(y_pred)), -1 * y_pred, 0, color = '0.25')
    ax.fill_between(np.arange(len(y_test)), 0, y_test, color = 'lightcoral')
    ax.set_yticks([-1,0,1])
    ax.set_yticklabels(['Predicted','Benign','Attacked'])
    plt.suptitle("")

    plt.tight_layout()  # Leaves space for figtext at the bottom    
    plt.show()