import torch
ckpt = torch.load("../models/checkpoint_final.pt", map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
print(type(state))
keys = list(state.keys())
print("Total keys:", len(keys))
print("First 40 keys:")
for k in keys[:40]:
    print(k)
