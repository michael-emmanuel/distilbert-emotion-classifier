import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix():
    preds = np.load("results/preds.npy")
    labels = np.load("results/labels.npy")
    
    # Emotion labels: sadness, joy, love, anger, fear, surprise
    target_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - DistilBERT Emotion Classifier')
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    plot_confusion_matrix()
