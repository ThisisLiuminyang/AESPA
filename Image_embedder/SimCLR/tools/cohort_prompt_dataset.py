from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CohortPromptDataset(Dataset):
    def __init__(self, patient_dict, topk_json, K, clinical_data_path, images_dir, verbose=True):
        self.K = K
        self.samples = []
        data = pd.read_excel(clinical_data_path)
        data["ID1"] = data["ID1"].astype(str).map(lambda value: value if value.startswith("CMMD_") else f"CMMD_{value}")
        if "Age" in data:
            data["age_"] = pd.qcut(data["Age"], 5, labels=False, duplicates="drop")
        data["label"] = data["classification"].astype(str).ne("Benign").astype(int)
        labels = data.set_index("ID1")["label"].to_dict()
        texts = {
            row.ID1: f"LeftRight:{getattr(row, 'LeftRight', '')} age_:{getattr(row, 'age_', '')} abnormality:{getattr(row, 'abnormality', '')}"
            for row in data.itertuples()
        }
        image_root = Path(images_dir)
        embeddings = {str(item["ID1"]): item["fusion_embeds"] for item in patient_dict}
        missing_base = missing_neighbors = 0
        for raw_pid, info in topk_json.items():
            pid = str(raw_pid)
            if pid not in embeddings:
                missing_base += 1
                continue
            base = embeddings[pid]
            if base.ndim == 1:
                base = base.unsqueeze(0)
            width = base.shape[-1]
            neighbors = []
            for neighbor_id in info.get("neighbors", [])[:K]:
                embedding = embeddings.get(str(neighbor_id))
                if embedding is None:
                    embedding = torch.zeros(1, width, dtype=base.dtype)
                    missing_neighbors += 1
                elif embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                neighbors.append(embedding)
            neighbors.extend(torch.zeros(1, width, dtype=base.dtype) for _ in range(K - len(neighbors)))
            neighbor_labels = list(info.get("neighbor_labels", [])[:K])
            neighbor_labels.extend([0] * (K - len(neighbor_labels)))
            pattern = f"{pid.removeprefix('CMMD_')}_*.png"
            self.samples.append({
                "med_embs": torch.cat([base, *neighbors]),
                "label": int(labels.get(pid, 0)),
                "neighbor_labels": neighbor_labels,
                "image_paths": sorted(image_root.glob(pattern)),
                "text": texts.get(pid, ""),
            })
        if verbose:
            print(f"patients={len(embeddings)} samples={len(self.samples)} missing_base={missing_base} missing_neighbors={missing_neighbors}")
        if not self.samples:
            raise RuntimeError("No valid dataset samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image_path = item["image_paths"][0] if item["image_paths"] else None
        image = Image.open(image_path).convert("RGB") if image_path else None
        return {
            "med_embs": item["med_embs"].clone().float(),
            "image": image,
            "image_path": str(image_path) if image_path else "",
            "text": item["text"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "neighbor_labels": torch.tensor(item["neighbor_labels"], dtype=torch.long),
        }
