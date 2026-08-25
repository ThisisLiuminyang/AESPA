from pathlib import Path

import pandas as pd
import torch
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT
from PIL import Image
from tqdm import tqdm


def to_patient_id_from_stem(stem):
    parts = stem.split("_")
    if stem.startswith("CMMD_") and len(parts) > 1:
        return "_".join(parts[:2])
    return f"CMMD_{parts[0]}"


def read_idx_as_patient_ids(idx_file):
    with Path(idx_file).open(encoding="utf-8") as file:
        return {to_patient_id_from_stem(line.strip()) for line in file if line.strip()}


def load_patient_dict(patient_dict_path, clinical_data_path, set_class, images_root, index_dir, model_dir, device="cuda", agg_mode="concat", concat_max_images=4, pad_concat=True):
    output_path = Path(patient_dict_path)
    if output_path.exists():
        return torch.load(output_path, map_location="cpu", weights_only=False)
    if agg_mode != "concat":
        raise ValueError("Only agg_mode='concat' is supported")
    data = pd.read_excel(clinical_data_path)
    data["ID1"] = data["ID1"].astype(str).map(lambda value: value if value.startswith("CMMD_") else f"CMMD_{value}")
    if "Age" in data:
        data["age_"] = pd.qcut(data["Age"], 5, labels=False, duplicates="drop")
    data["label"] = data["classification"].astype(str).ne("Benign").astype(int)
    rows = data.set_index("ID1")
    selected_ids = read_idx_as_patient_ids(Path(index_dir) / f"idx_{set_class}")
    image_paths = [path for path in Path(images_root).glob("*.png") if to_patient_id_from_stem(path.stem) in selected_ids and to_patient_id_from_stem(path.stem) in rows.index]
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(input_dir=str(model_dir))
    model = model.to(device).eval()
    cache = {}
    with torch.no_grad():
        for image_path in tqdm(sorted(image_paths)):
            pid = to_patient_id_from_stem(image_path.stem)
            row = rows.loc[pid]
            text = f"LeftRight:{row.get('LeftRight', '')} age_:{row.get('age_', '')} abnormality:{row.get('abnormality', '')}"
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
            output = model(**inputs)
            embedding = torch.cat([output["img_embeds"], output["text_embeds"]], dim=1).cpu()
            record = cache.setdefault(pid, {"feats": [], "label": int(row["label"]), "image_paths": [], "text": text})
            record["feats"].append(embedding)
            record["image_paths"].append(str(image_path))
    patients = []
    for pid, record in cache.items():
        features = record["feats"][:concat_max_images]
        if pad_concat:
            features.extend(torch.zeros_like(features[0]) for _ in range(concat_max_images - len(features)))
        patients.append({
            "ID1": pid,
            "fusion_embeds": torch.cat(features).reshape(1, -1),
            "label": record["label"],
            "image_paths": record["image_paths"][:concat_max_images],
            "text": record["text"],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(patients, output_path)
    return patients
