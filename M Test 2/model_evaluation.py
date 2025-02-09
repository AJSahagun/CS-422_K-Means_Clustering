# model_evaluation.py
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from scipy.optimize import linear_sum_assignment

def map_clusters_to_labels(true_labels, cluster_labels):
    """Hungarian algorithm for label mapping"""
    contingency_matrix = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return col_ind[np.argsort(row_ind)]

def evaluate_clustering(true_labels, cluster_labels):
    """Calculate evaluation metrics"""
    mapped_labels = map_clusters_to_labels(true_labels, cluster_labels)
    adjusted_labels = mapped_labels[cluster_labels]
    
    return {
        'confusion_matrix': confusion_matrix(true_labels, adjusted_labels),
        'precision': precision_score(true_labels, adjusted_labels, average='weighted'),
        'recall': recall_score(true_labels, adjusted_labels, average='weighted'),
        'f1_score': f1_score(true_labels, adjusted_labels, average='weighted')
    }

# Validation test (Member B should create small test data)
def test_metrics():
    """Test with known small dataset"""
    test_labels = np.array([0,0,1,1,2,2])
    test_clusters = np.array([1,1,0,0,2,2])
    results = evaluate_clustering(test_labels, test_clusters)
    assert np.all(results['confusion_matrix'] == [[2,0,0],[0,2,0],[0,0,2]]), "Validation failed"