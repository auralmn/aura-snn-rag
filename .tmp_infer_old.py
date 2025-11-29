import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer

# Config matching old training script
class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embedding_dim = 768
        self.num_layers = 12
        self.num_heads = 12
        self.dropout = 0.1
        self.max_seq_len = 512
        self.intermediate_size = 3072
        self.use_snn_ffn = True
        self.snn_layers = [0,2,4,6,8,10]
        self.use_rag = True
        self.num_retrieved = 3

class SimpleHippocampus(nn.Module):
    def __init__(self, feature_dim, max_memories=100000, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_memories = max_memories
        self.device = device
        self.memory_count = 0
        self.register_buffer('memory_features', torch.zeros(max_memories, feature_dim))
        self.register_buffer('memory_strengths', torch.ones(max_memories))
    def store(self, features):
        if self.memory_count >= self.max_memories:
            idx = self.memory_count % self.max_memories
        else:
            idx = self.memory_count
            self.memory_count += 1
        feat = features.detach().mean(dim=0) if features.dim() > 1 else features.detach()
        self.memory_features[idx] = feat
        self.memory_strengths[idx] = 1.0
    def retrieve(self, query, k=5):
        if self.memory_count == 0:
            return torch.zeros(k, self.feature_dim, device=self.memory_features.device), torch.zeros(k, device=self.memory_features.device)
        k = min(k, self.memory_count)
        active = self.memory_features[:self.memory_count]
        q_norm = F.normalize(query.unsqueeze(0), dim=1)
        m_norm = F.normalize(active, dim=1)
        scores = torch.mm(q_norm, m_norm.t()).squeeze(0)
        top_scores, top_idx = torch.topk(scores, k)
        return self.memory_features[top_idx], top_scores
    def decay(self, rate=0.01):
        self.memory_strengths *= (1.0 - rate)

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, L):
        ctx.save_for_backward(x)
        return torch.floor(torch.clamp(x, 0, L))
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = torch.sigmoid(x) * (1 - torch.sigmoid(x)) * 4
        return grad_output * grad, None

class GIFNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, L=8, dt=1.0, tau=10.0, threshold=1.0, alpha=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L = L
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.decay = math.exp(-dt / tau)
        self.threshold = threshold
        self.alpha = alpha
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype
        if state is None:
            v = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            theta = torch.full((batch_size, self.hidden_dim), self.threshold, device=device, dtype=dtype)
        else:
            v, theta = state
        h = self.linear(x)
        spikes_list = []
        for t in range(seq_len):
            i_t = h[:, t, :]
            v = v * self.decay + i_t
            v = torch.clamp(v, -self.L * theta * 2, self.L * theta * 2)
            normalized_v = v / (theta + 1e-6)
            spike = SurrogateSpike.apply(normalized_v, self.L)
            v = v - spike * theta
            if self.alpha > 0:
                theta = theta + self.alpha * spike - self.alpha * (theta - self.threshold)
            spikes_list.append(spike)
        return torch.stack(spikes_list, dim=1), (v, theta)

class Synapsis(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        std = 1.0 / math.sqrt(in_features * 0.3)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.zeros_(self.bias)
    def forward(self, spikes, state=None):
        batch_size, seq_len, _ = spikes.shape
        spikes_flat = spikes.reshape(batch_size * seq_len, -1)
        currents_flat = F.linear(spikes_flat, self.weight, self.bias)
        return currents_flat.reshape(batch_size, seq_len, -1), None

class SNNFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_timesteps=4, L=8, dropout=0.1):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.syn1 = Synapsis(input_dim, hidden_dim)
        self.neuron1 = GIFNeuron(hidden_dim, hidden_dim, L=L)
        self.syn2 = Synapsis(hidden_dim, input_dim)
        self.neuron2 = GIFNeuron(input_dim, input_dim, L=L)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.num_timesteps, -1)
        x_flat = x_expanded.reshape(batch_size * seq_len, self.num_timesteps, dim)
        h1, _ = self.syn1(x_flat, state=None)
        spikes1, _ = self.neuron1(h1, state=None)
        h2, _ = self.syn2(spikes1, state=None)
        spikes2, _ = self.neuron2(h2, state=None)
        output_flat = spikes2.mean(dim=1)
        return self.dropout(output_flat.reshape(batch_size, seq_len, -1))

class HybridFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, snn_ratio=0.5, dropout=0.1):
        super().__init__()
        self.snn = SNNFFN(input_dim, hidden_dim, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.snn_ratio = snn_ratio
    def forward(self, x):
        return self.snn_ratio * self.snn(x) + (1 - self.snn_ratio) * self.mlp(x)

class TransformerLayer(nn.Module):
    def __init__(self, config, use_snn=False, hippocampus=None):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        self.attn_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.embedding_dim)
        if use_snn:
            self.ffn = HybridFFN(config.embedding_dim, config.intermediate_size, dropout=config.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.embedding_dim, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.embedding_dim),
                nn.Dropout(config.dropout),
            )
        self.dropout = nn.Dropout(config.dropout)
        if config.use_rag and hippocampus is not None:
            self.memory_gate = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                nn.Sigmoid(),
            )
            self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
    def forward(self, x, use_memory=True, store_memory=False):
        normed = self.attn_norm(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=causal_mask, is_causal=True)
        x = x + self.dropout(attn_out)
        if use_memory and self.hippocampus is not None and self.config.use_rag:
            if self.hippocampus.memory_count > 0:
                query = self.query_proj(x.mean(dim=1))
                mem_features, mem_scores = self.hippocampus.retrieve(query[0], k=self.config.num_retrieved)
                weights = F.softmax(mem_scores, dim=-1).unsqueeze(-1)
                mem_context = (mem_features * weights).sum(dim=0, keepdim=True)
                mem_context = mem_context.unsqueeze(0).expand(x.size(0), x.size(1), -1)
                gate_input = torch.cat([x, mem_context], dim=-1)
                gate = self.memory_gate(gate_input)
                x = x + gate * mem_context
        normed = self.ffn_norm(x)
        x = x + self.ffn(normed)
        if store_memory and self.hippocampus is not None:
            self.hippocampus.store(x.mean(dim=1)[0])
        return x

class AuraTransformer(nn.Module):
    def __init__(self, config, hippocampus):
        super().__init__()
        self.config = config
        self.hippocampus = hippocampus
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        snn_layers = set(config.snn_layers) if config.snn_layers else set()
        self.layers = nn.ModuleList([
            TransformerLayer(config, use_snn=(config.use_snn_ffn and i in snn_layers), hippocampus=hippocampus)
            for i in range(config.num_layers)
        ])
        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
    def forward(self, input_ids, use_memory=True, store_memory=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, use_memory=use_memory, store_memory=store_memory)
        logits = self.output_head(x)
        return logits


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Config()
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base', legacy=True, local_files_only=True)
    cfg.vocab_size = tokenizer.vocab_size
    hippocampus = SimpleHippocampus(feature_dim=cfg.embedding_dim, max_memories=100000, device=str(device)).to(device)
    model = AuraTransformer(cfg, hippocampus).to(device)
    sd = torch.load('../models/checkpoint_final.pt', map_location=device)
    state = sd.get('model_state_dict', sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    prompt = "Aura is a hybrid ANN-SNN model designed for energy-efficient reasoning."
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    model.eval()
    max_new_tokens = 40
    with torch.no_grad():
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = model(generated, use_memory=False, store_memory=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                break
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print('\nPrompt:', prompt)
    print('Output:', text)

if __name__ == '__main__':
    main()
