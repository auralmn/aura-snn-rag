import torch
sd = torch.load('../models/checkpoint_final.pt', map_location='cpu')
state = sd.get('model_state_dict', sd)
print('token_embedding', state['token_embedding.weight'].shape)
print('pos_embedding', state['pos_embedding.weight'].shape)
heads = [k for k in state if k.endswith('attention.in_proj_weight')]
print('num layers', len(heads))
print('attention in_proj shape first', state[heads[0]].shape)
print('ffn0', state['layers.0.ffn.mlp.0.weight'].shape)
print('hippocampus memory_features', state['hippocampus.memory_features'].shape)
