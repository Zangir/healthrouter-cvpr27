#!/usr/bin/env python3
"""BiomedCLIP text embeddings for the zero-shot modality gate (the earlier paper's prompts,
plus two prompts for the non-medical tasks). Writes cache/router/biomedclip_text.npz."""
import numpy as np
import open_clip
import torch

from healthrouter.collection import data

PROMPTS = {"chest_xray": "a chest X-ray radiograph",
           "pathology": "an H&E stained histopathology patch",
           "fundus": "a color fundus photograph of the retina",
           "mammography": "a mammogram",
           "eurosat": "a satellite image of land use",
           "dtd": "a close-up photograph of a texture"}
name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, _ = open_clip.create_model_from_pretrained(name)
tok = open_clip.get_tokenizer(name)
with torch.no_grad():
    t = model.encode_text(tok(list(PROMPTS.values()))).float()
    t = (t / t.norm(dim=-1, keepdim=True)).numpy()
out = data.cache_dir() / "router" / "biomedclip_text.npz"
np.savez(out, emb=t, tasks=np.array(list(PROMPTS)), prompts=np.array(list(PROMPTS.values())))
print(out, t.shape)
