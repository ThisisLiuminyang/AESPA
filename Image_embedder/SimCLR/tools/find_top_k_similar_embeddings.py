import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def find_top_k_similar_embeddings_cross_split(query_patients, retrieval_patients, top_k_json_path, K, set_class, dynamic_retrieval=False):
    output_path = Path(top_k_json_path)
    if output_path.exists() and not dynamic_retrieval:
        print(f"{set_class} Top-{K} retrieval already exists: {output_path}")
        return None

    def unpack(patients):
        ids, labels, vectors = [], [], []
        for item in patients:
            embedding = item["fusion_embeds"]
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            embedding = np.asarray(embedding)
            if embedding.ndim == 2:
                embedding = embedding.mean(axis=0)
            ids.append(str(item["ID1"]))
            labels.append(int(item.get("label", -1)))
            vectors.append(embedding.astype(np.float32).reshape(-1))
        if not vectors:
            raise ValueError("Patient data cannot be empty")
        return np.asarray(ids), np.asarray(labels), np.stack(vectors)

    query_ids, query_labels, query_vectors = unpack(query_patients)
    retrieval_ids, retrieval_labels, retrieval_vectors = unpack(retrieval_patients)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True).clip(min=1e-12)
    retrieval_vectors /= np.linalg.norm(retrieval_vectors, axis=1, keepdims=True).clip(min=1e-12)
    similarities = query_vectors @ retrieval_vectors.T
    result = {}
    for index in tqdm(range(len(query_ids)), desc=f"{set_class} retrieval"):
        row = similarities[index].copy()
        row[retrieval_ids == query_ids[index]] = -np.inf
        count = min(max(int(K), 0), len(row))
        indices = np.argsort(-row)[:count]
        result[query_ids[index]] = {
            "neighbors": retrieval_ids[indices].tolist(),
            "similarities": row[indices].astype(float).tolist(),
            "neighbor_labels": retrieval_labels[indices].astype(int).tolist(),
            "self_label": int(query_labels[index]),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result
