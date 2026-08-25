import sys

sys.path.append("/home/icdm/lmy/Multimodal-Medical-master/LLaVA")
import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

class FrozenQwenPromptClassifier(nn.Module):
    """
    Fixed & Stable Frozen Qwen3-VL classifier

    Fixes:
    - correct soft prompt usage (no mean collapse)
    - correct pooling (last valid token)
    - neighbor labels as ratio (NO label leakage)
    - dtype-safe (bf16 compatible)
    """

    def __init__(
        self,
        qwen_model,
        processor,
        num_labels,
    ):
        super().__init__()

        self.qwen = qwen_model
        self.processor = processor
        self.num_labels = num_labels

        self.hidden = qwen_model.config.text_config.hidden_size

        # lazy init
        self.prompt_proj = None

        # 🔥 classifier input dim (NO one-hot!)
        cls_dim = self.hidden
        self.classifier = nn.Linear(cls_dim, num_labels)

        # 🔒 freeze Qwen
        for p in self.qwen.parameters():
            p.requires_grad = False

    def forward(
        self,
        prompt_vectors,  # [B, P, dim_medclip]
        images,  # list[PIL.Image]
        texts,  # list[str]
    ):
        device = prompt_vectors.device
        B, P, _ = prompt_vectors.shape

        qwen_dtype = next(self.qwen.parameters()).dtype

        if self.prompt_proj is None:
            self.prompt_proj = (
                nn.Linear(prompt_vectors.size(-1), self.hidden, bias=False)
                .to(device)
                .to(qwen_dtype)
            )

        prompt_tokens = self.prompt_proj(prompt_vectors.to(qwen_dtype))  # [B,P,H]

        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(device)

        # embeddings
        text_embeds = self.qwen.get_input_embeddings()(inputs["input_ids"])  # [B,T,H]

        # prepend soft prompt
        inputs_embeds = torch.cat([prompt_tokens, text_embeds], dim=1)  # [B,P+T,H]

        prompt_mask = torch.ones(
            B, P, device=device, dtype=inputs["attention_mask"].dtype
        )

        attention_mask = torch.cat([prompt_mask, inputs["attention_mask"]], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        seq_len = attention_mask.sum(dim=1) - 1  # [B]

        pooled = outputs.hidden_states[-1][
            torch.arange(B, device=device), seq_len
        ]  # [B,H]

        logits = self.classifier(pooled.float())

        return logits, pooled