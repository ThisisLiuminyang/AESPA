from pathlib import Path

from find_top_k_similar_embeddings import find_top_k_similar_embeddings_cross_split


def read_top_k_json(query_patients, retrieval_patients, top_k_json_path, K, set_class, dynamic_retrieval=False):
    path = Path(top_k_json_path)
    if path.exists() and not dynamic_retrieval:
        print(f"{set_class} Top-{K} data already exists: {path}")
        return None
    return find_top_k_similar_embeddings_cross_split(query_patients, retrieval_patients, path, K, set_class, dynamic_retrieval)
