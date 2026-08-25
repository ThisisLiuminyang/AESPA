import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Image_embedder.SimCLR.tools.load_patient_dict import (
    load_patient_dict,
)
from Image_embedder.SimCLR.tools.read_top_k_json import (
    read_top_k_json,
)
from MultiplexNetwork.models.model_attention_pooling import (
    CohortEvidenceAttentionPooling,
)
from MultiplexNetwork.models.model_frozen_qwen import FrozenQwenPromptClassifier
from Image_embedder.SimCLR.tools.dataset.cohort_prompt_dataset import (
    CohortPromptDataset,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", default="CMMD")
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--dynamic-retrieval", action="store_true")
    parser.add_argument("--use-caption", action="store_true")
    parser.add_argument("--use-contrastive-learning", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--evidence-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.05)
    parser.add_argument("--evaluation-repeats", type=int, default=50)
    parser.add_argument("--linear-probe-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_suffix(args):
    if args.use_caption and args.use_contrastive_learning:
        return "patients_dict_with_caption_and_contrastive_learning.pt"
    if args.use_caption:
        return "patients_dict_with_caption.pt"
    if args.use_contrastive_learning:
        return "patients_dict_with_contrastive_learning.pt"
    return "patients_dict.pt"


def collate_batch(batch):
    return {
        "med_embs": torch.stack([item["med_embs"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "neighbor_labels": torch.stack([item["neighbor_labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
        "images": [item["image"] for item in batch],
    }


def build_loaders(args, device):
    root = args.dataset_root / args.data
    clinical_path = root / f"{args.data}_clinicaldata_revision.xlsx"
    suffix = patient_suffix(args)
    patients = {
        split: load_patient_dict(
            patient_dict_path=str(root / "patient_dict" / f"{split}_{suffix}"),
            clinical_data_path=str(clinical_path),
            use_caption=args.use_caption,
            set_class=split,
            device=device,
        )
        for split in ("train", "val", "test")
    }
    loaders = {}
    with tempfile.TemporaryDirectory() as directory:
        for split in ("train", "val", "test"):
            path = Path(directory) / f"{split}.json"
            read_top_k_json(
                query_patients=patients[split],
                retrieval_patients=patients["train"],
                top_k_json_path=str(path),
                K=args.k,
                set_class=split,
                dynamic_retrieval=args.dynamic_retrieval,
            )
            with path.open(encoding="utf-8") as file:
                neighbors = json.load(file)
            dataset = CohortPromptDataset(
                patient_dict=patients[split], topk_json=neighbors, K=args.k
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                collate_fn=collate_batch,
            )
    return loaders


def evidence_loss(attention, neighbor_labels):
    if attention.size(1) == neighbor_labels.size(1) + 1:
        attention = attention[:, 1:]
    if attention.shape != neighbor_labels.shape:
        raise ValueError("Attention and neighbor-label shapes must match")
    attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
    targets = neighbor_labels.float()
    targets = targets / targets.sum(1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(attention.clamp_min(1e-12).log(), targets, reduction="batchmean")


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature
        labels = labels.view(-1, 1)
        diagonal = 1 - torch.eye(features.size(0), device=features.device)
        positives = labels.eq(labels.T).float() * diagonal
        log_prob = logits - torch.log(
            (torch.exp(logits) * diagonal).sum(1, keepdim=True).clamp_min(1e-12)
        )
        mean_positive = (positives * log_prob).sum(1) / positives.sum(1).clamp_min(
            1e-12
        )
        return (-(self.temperature / self.base_temperature) * mean_positive).mean()


def forward_batch(batch, model_llm, model_attn, device):
    embeddings = batch["med_embs"].to(device)
    labels = batch["labels"].to(device)
    neighbor_labels = batch["neighbor_labels"].to(device)
    prompts, attention = model_attn(embeddings)
    logits, pooled = model_llm(
        prompt_vectors=prompts,
        images=batch["images"],
        texts=batch["texts"],
        neighbor_labels=neighbor_labels,
    )
    return logits, pooled, attention, labels, neighbor_labels


def calculate_loss(outputs, cross_entropy, contrastive, args):
    logits, pooled, attention, labels, neighbor_labels = outputs
    return (
        cross_entropy(logits, labels)
        + args.contrastive_weight * contrastive(pooled, labels)
        + args.evidence_weight * evidence_loss(attention, neighbor_labels)
    )


def validation_accuracy(loader, model_llm, model_attn, device):
    model_llm.eval()
    model_attn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits, _, _, labels, _ = forward_batch(
                batch, model_llm, model_attn, device
            )
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def extract_embeddings(loader, model_llm, model_attn, device):
    model_llm.eval()
    model_attn.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            _, pooled, _, batch_labels, _ = forward_batch(
                batch, model_llm, model_attn, device
            )
            embeddings.append(pooled.float().cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def evaluate(loaders, model_llm, model_attn, device, args):
    train_x, train_y = extract_embeddings(
        loaders["train"], model_llm, model_attn, device
    )
    val_x, val_y = extract_embeddings(loaders["val"], model_llm, model_attn, device)
    test_x, test_y = extract_embeddings(loaders["test"], model_llm, model_attn, device)
    class_count = torch.unique(torch.cat((train_y, val_y, test_y))).numel()
    metrics = {
        "accuracy": [],
        "macro_f1": [],
        "micro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
    }
    for repeat in range(args.evaluation_repeats):
        torch.manual_seed(args.seed + repeat)
        classifier = nn.Linear(train_x.size(1), class_count).to(device)
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=0.01, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        best_validation = -1.0
        selected = None
        for _ in range(args.linear_probe_epochs):
            classifier.train()
            optimizer.zero_grad()
            criterion(classifier(train_x.to(device)), train_y.to(device)).backward()
            optimizer.step()
            classifier.eval()
            with torch.no_grad():
                val_predictions = classifier(val_x.to(device)).argmax(1).cpu()
                score = f1_score(
                    val_y.numpy(),
                    val_predictions.numpy(),
                    average="macro",
                    zero_division=0,
                )
                if score > best_validation:
                    best_validation = score
                    selected = classifier(test_x.to(device)).argmax(1).cpu().numpy()
        truth = test_y.numpy()
        metrics["accuracy"].append(accuracy_score(truth, selected))
        metrics["macro_f1"].append(
            f1_score(truth, selected, average="macro", zero_division=0)
        )
        metrics["micro_f1"].append(
            f1_score(truth, selected, average="micro", zero_division=0)
        )
        metrics["macro_precision"].append(
            precision_score(truth, selected, average="macro", zero_division=0)
        )
        metrics["macro_recall"].append(
            recall_score(truth, selected, average="macro", zero_division=0)
        )
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in metrics.items()
    }


def initialize_models(args, device):
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    qwen_model.eval()
    for parameter in qwen_model.parameters():
        parameter.requires_grad = False
    model_attn = CohortEvidenceAttentionPooling(
        dim_medclip=args.embedding_dim,
        dim_qwen=qwen_model.language_model.config.hidden_size,
        prompt_len=args.prompt_length,
    ).to(device)
    model_llm = FrozenQwenPromptClassifier(
        qwen_model=qwen_model,
        processor=processor,
        num_labels=2,
    ).to(device)
    return model_llm, model_attn


def train(args, loaders, model_llm, model_attn, device):
    optimizer = torch.optim.AdamW(
        list(model_attn.parameters()) + list(model_llm.classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    cross_entropy = nn.CrossEntropyLoss()
    contrastive = SupervisedContrastiveLoss(args.contrastive_temperature).to(device)
    best_accuracy = -1.0
    best_attention = None
    best_classifier = None
    for _ in range(args.epochs):
        model_attn.train()
        model_llm.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            outputs = forward_batch(batch, model_llm, model_attn, device)
            calculate_loss(outputs, cross_entropy, contrastive, args).backward()
            optimizer.step()
        accuracy = validation_accuracy(loaders["val"], model_llm, model_attn, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_attention = copy.deepcopy(model_attn.state_dict())
            best_classifier = copy.deepcopy(model_llm.classifier.state_dict())
    model_attn.load_state_dict(best_attention)
    model_llm.classifier.load_state_dict(best_classifier)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    loaders = build_loaders(args, device)
    model_llm, model_attn = initialize_models(args, device)
    train(args, loaders, model_llm, model_attn, device)
    return evaluate(loaders, model_llm, model_attn, device, args)


if __name__ == "__main__":
    main()
