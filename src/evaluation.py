def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant = set(relevant)
    return len(set(recommended) & relevant) / k

def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant = set(relevant)
    return len(set(recommended) & relevant) / len(relevant)