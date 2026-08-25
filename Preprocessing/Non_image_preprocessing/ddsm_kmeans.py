import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale


FIELDS = [
    "breast_density",
    "left or right breast",
    "abnormality type",
    "mass shape",
    "mass margins",
    "assessment",
]
FIELD_GROUPS = [
    ["breast_density"],
    ["left or right breast", "abnormality type"],
    ["mass shape", "mass margins"],
    ["assessment"],
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--clinical-train", type=Path, required=True)
    parser.add_argument("--clinical-test", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs=4, default=(0.9, 0.93, 0.75, 0.5)
    )
    parser.add_argument("--nearest-neighbors", type=int, default=3)
    return parser.parse_args()


def decode_image_id(value):
    digits = str(int(float(value)))
    prefix = digits[:-5]
    if not prefix:
        raise ValueError("Image ID cannot be converted to a patient ID")
    return f"P_{prefix.zfill(5)}"


def load_clinical(train_path, test_path):
    data = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True
    )
    pathology = data["pathology"].astype(str).str.upper()
    data["label"] = np.select(
        [pathology.str.contains("MALIGNANT"), pathology.str.contains("BENIGN")],
        [1, 0],
        default=-1,
    )
    data = data[data["label"] >= 0].copy()
    return data.drop_duplicates("patient_id", keep="first").reset_index(drop=True)


def align_features(feature_path, id_path, clinical):
    features = np.loadtxt(feature_path, delimiter=",", dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    raw_ids = pd.read_csv(id_path, header=None)[0].values
    ids = [decode_image_id(value) for value in raw_ids]
    if len(ids) != len(features):
        raise ValueError("Image ID and feature counts differ")
    valid_ids = set(clinical["patient_id"])
    positions = [index for index, patient_id in enumerate(ids) if patient_id in valid_ids]
    if not positions:
        raise ValueError("No image features match the clinical records")
    return features[positions], [ids[index] for index in positions]


def encode_clinical(clinical):
    encoded = pd.get_dummies(clinical[FIELDS], columns=FIELDS).fillna(0)
    features = {
        patient_id: encoded.iloc[index].to_numpy(dtype=np.float32)
        for index, patient_id in enumerate(clinical["patient_id"])
    }
    labels = {
        patient_id: int(clinical.iloc[index]["label"])
        for index, patient_id in enumerate(clinical["patient_id"])
    }
    return features, labels


def build_adjacency(clinical, patient_ids, fields, threshold, neighbor_count):
    encoded = pd.get_dummies(clinical[fields], columns=fields).fillna(0)
    row_by_id = {
        patient_id: encoded.iloc[index].to_numpy(dtype=np.float32)
        for index, patient_id in enumerate(clinical["patient_id"])
    }
    matrix = np.stack([row_by_id[patient_id] for patient_id in patient_ids])
    matrix = minmax_scale(matrix, axis=0, copy=True)
    similarity = cosine_similarity(matrix, matrix)
    adjacency = (similarity > threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    isolated = np.flatnonzero(adjacency.sum(axis=1) == 1)
    available = max(len(patient_ids) - 1, 0)
    count = min(neighbor_count, available)
    if count:
        for index in isolated:
            scores = similarity[index].copy()
            scores[index] = -np.inf
            neighbors = np.argpartition(scores, -count)[-count:]
            adjacency[index, neighbors] = 1.0
            adjacency[neighbors, index] = 1.0
    return adjacency


def load_indices(path, position_by_id):
    with path.open(encoding="utf-8") as file:
        patient_ids = [line.strip() for line in file if line.strip()]
    return np.asarray(
        [position_by_id[patient_id] for patient_id in patient_ids if patient_id in position_by_id],
        dtype=np.int32,
    )


def build_dataset(args):
    clinical = load_clinical(args.clinical_train, args.clinical_test)
    image_features, patient_ids = align_features(
        args.image_features, args.image_ids, clinical
    )
    clinical_features, label_by_id = encode_clinical(clinical)
    features = np.stack(
        [
            np.concatenate((image_features[index], clinical_features[patient_id]))
            for index, patient_id in enumerate(patient_ids)
        ]
    ).astype(np.float32)
    labels = np.asarray([label_by_id[patient_id] for patient_id in patient_ids])
    one_hot_labels = np.eye(2, dtype=np.float32)[labels]
    adjacency = [
        build_adjacency(
            clinical,
            patient_ids,
            fields,
            threshold,
            args.nearest_neighbors,
        )
        for fields, threshold in zip(FIELD_GROUPS, args.thresholds)
    ]
    positions = {}
    for index, patient_id in enumerate(patient_ids):
        positions.setdefault(patient_id, index)
    result = {
        "feature": features,
        "label": one_hot_labels,
        "train_idx": load_indices(args.train_index, positions),
        "val_idx": load_indices(args.validation_index, positions),
        "test_idx": load_indices(args.test_index, positions),
    }
    result.update({f"type{index}": matrix for index, matrix in enumerate(adjacency)})
    return result


def main():
    args = parse_args()
    dataset = build_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
    return dataset


if __name__ == "__main__":
    main()
