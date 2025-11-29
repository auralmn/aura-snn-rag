import torch
from transformers import T5Tokenizer
from colab_l4_training import get_full_config
from core.hippocampal import HippocampalFormation
from core.language_zone.hippocampal_transformer import HippocampalTransformer

ckpt_path = "../models/checkpoint_final.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

cfg = get_full_config()
cfg.checkpoint_path = ckpt_path
cfg.use_gradient_checkpointing = False
cfg.enable_centroid_index = True

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
cfg.vocab_size = tokenizer.vocab_size

print("Building hippocampus/model...")
hippocampus = HippocampalFormation(
    feature_dim=cfg.embedding_dim,
    n_place_cells=cfg.n_place_cells,
    n_time_cells=cfg.n_time_cells,
    n_grid_cells=cfg.n_grid_cells,
    max_memories=cfg.max_memories,
    device=str(device),
    use_centroid_index=cfg.enable_centroid_index,
).to(device)
model = HippocampalTransformer(cfg, hippocampus).to(device)

print(f"Loading checkpoint from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt.get('model_state_dict', ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print("Missing sample:", missing[:5])
if unexpected:
    print("Unexpected sample:", unexpected[:5])

model.eval()

prompt = "Aura is a hybrid ANN-SNN model designed for energy-efficient reasoning."
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
max_new_tokens = 40
print("Running generation...")
with torch.no_grad():
    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -cfg.max_seq_len:]
        logits, _ = model(context, use_memory=False, store_memory=False)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
            break

text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print("\nPrompt:", prompt)
print("Output:", text)
