import numpy as np
import json
import os

class SimilarCaseRetrieval:
    """
    Content-Based Image Retrieval (CBIR) to find similar historical cases.
    """
    def __init__(self, db_path=None):
        self.embeddings = None
        self.metadata = None
        
        if db_path and os.path.exists(db_path):
            self.load_database(db_path)

    def load_database(self, db_path):
        """
        Load embeddings and case metadata.
        """
        with open(db_path, 'r') as f:
            db = json.load(f)
        self.embeddings = np.array(db["embeddings"], dtype=np.float32)
        self.metadata = db["metadata"]
        
        # Normalize embeddings for fast cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings_normalized = self.embeddings / norms

    def retrieve(self, query_embedding, k=3):
        """
        Find top-k similar cases.
        """
        if self.embeddings is None or self.metadata is None:
            return []
            
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        query_normalized = query_embedding / query_norm if query_norm > 0 else query_embedding
        
        # Compute cosine similarities via dot product
        similarities = np.dot(self.embeddings_normalized, query_normalized)
        
        # Get top-k indices (sorted descending)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            case_meta = self.metadata[idx]
            
            results.append({
                "patient_id": case_meta["patient_id"],
                "filename": case_meta["filename"],
                "label": int(case_meta["label"]),
                "similarity": score,
                "age": case_meta["age"],
                "gender": case_meta["gender"],
                "temperature": case_meta["temperature"],
                "spo2": case_meta["spo2"]
            })
            
        return results

    def save_database(self, db_path, embeddings, metadata):
        """
        Build and save the retrieval database.
        """
        db = {
            "embeddings": embeddings.tolist(),
            "metadata": metadata
        }
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=2)
