import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import numpy as np 
import math
import pdb
try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False



num_atom_type = 119  # including the extra mask molecules
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    """GINE message passing layer for molecular graphs."""

    def __init__(self, emb_dim):
        """Initialize the GINE layer with edge embeddings."""
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        """Apply GINE message passing with self-loop handling."""
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """Combine neighbor messages with edge embeddings."""
        return x_j + edge_attr

    def update(self, aggr_out):
        """Project aggregated messages with an MLP."""
        return self.mlp(aggr_out)




class Conditional_AGILE(nn.Module): # Conditional Molecule encoder (CME). If condition is None, it becomes the standard Molecule encoder (ME).
    """GNN backbone for molecular graphs with optional per-graph conditioning."""

    def __init__(self, args):
        super(Conditional_AGILE, self).__init__()
        self.args = args
        self.num_layers = getattr(args, "num_layers", 5)
        self.emb_dim = 300
        self.hidden = args.hidden
        self.drop = args.drop

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList([GINEConv(self.emb_dim) for _ in range(self.num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.emb_dim) for _ in range(self.num_layers)])
        self.adaLM_scale = nn.ModuleList([nn.Linear(self.hidden, self.emb_dim) for _ in range(self.num_layers)])
        self.adaLM_shift = nn.ModuleList([nn.Linear(self.hidden, self.emb_dim) for _ in range(self.num_layers)])

        self.pool = global_mean_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.hidden)

    def forward(self, data, condition=None):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if condition is not None:
            if condition.dim() != 2 or condition.size(0) != data.num_graphs:
                raise ValueError(
                    f"Expected condition shape [num_graphs, hidden], got {tuple(condition.shape)}."
                )
            node_condition = condition[data.batch]
        else:
            node_condition = None

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layers):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if node_condition is not None:
                h = h * (1 + self.adaLM_scale[layer](node_condition)) + self.adaLM_shift[layer](node_condition)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        return h

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            if own_state[name].shape != param.shape:
                continue
            own_state[name].copy_(param)



def rotate_half(x): # degree 90 rotation operation
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, device=None, learnable_base=False):
        super().__init__()
        self.dim = dim
        self.learnable_base = learnable_base
        if self.learnable_base:
            self.base_param = torch.nn.Parameter(torch.tensor(float(base), dtype=torch.float32))
            self.register_buffer("inv_freq", None, persistent=False)
        else:
            self.base = float(base)
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_parameters(self):
        if self.learnable_base:
            with torch.no_grad():
                self.base_param.fill_(10000.0)

    def forward(self, position_ids, dtype=None):
        if dtype is None:
            dtype = position_ids.dtype
        if self.learnable_base:
            base = 1.0 + F.softplus(self.base_param)
            inv_freq = 1.0 / ( base ** (torch.arange(0, self.dim, 2, device=position_ids.device, dtype=self.base_param.dtype) / self.dim) )
        else:
            inv_freq = self.inv_freq.to(device=position_ids.device)
        position_ids = position_ids.to(device=position_ids.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("bs,d->bsd", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class interaction_transformer(torch.nn.Module):    # STRATA
    def __init__(self, args):
        super(interaction_transformer, self).__init__()
        self.args = args
        self.num_molecules = args.num_molecules
        self.emb_dim = args.hidden
        self.dropout = args.dropout
        self.ffn_dim = self.emb_dim * 4
        self.pos_enc = args.pos_enc
        if self.pos_enc in ['RoPE', 'RiPE']:
            self.use_rotary = True
        else: 
            self.use_rotary = False
        self.molecule_encoder = Conditional_AGILE(args)             # Molecule Encoder (ME)
        self.learnable_molecule_encoder = Conditional_AGILE(args)   # Conditional Molecule Encoder (CME)
        
        self.layers = torch.nn.ModuleList([
            _InteractionTransformerLayer(args, self.emb_dim, self.ffn_dim, self.dropout, self.use_rotary)
            for _ in range(self.args.num_layers)
        ])
        self.RNA_token = torch.nn.Parameter(torch.zeros(self.emb_dim))
        self.CLS_token = torch.nn.Parameter(torch.zeros(self.emb_dim))
        self.rand_pos_emb = torch.nn.Parameter(torch.zeros(self.num_molecules, self.emb_dim))
        self.final_norm = torch.nn.LayerNorm(self.emb_dim)
        
        regression_in_dim = self.emb_dim * self.num_molecules
        self.regression_mlp = torch.nn.Sequential(
            torch.nn.Linear(regression_in_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, 1),
        )
        self.regression_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(regression_in_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, 1),
        )
        
        self.emb_dropout = torch.nn.Dropout(self.dropout)
        self.reset_parameters()
        return
    

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.RNA_token.view(1, -1))
        torch.nn.init.xavier_uniform_(self.CLS_token.view(1, -1))
        torch.nn.init.xavier_uniform_(self.rand_pos_emb)
        self.final_norm.reset_parameters()
        for layer in self.layers:
            layer.attn_norm.reset_parameters()
            layer.attn.qkv.reset_parameters()
            layer.attn.out_proj.reset_parameters()
            if layer.attn.rotary is not None and hasattr(layer.attn.rotary, "reset_parameters"):
                layer.attn.rotary.reset_parameters()
            layer.ffn_norm.reset_parameters()
            for module in layer.ffn:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                                          
        self.molecule_encoder = Conditional_AGILE(self.args).to(self.args.device)
        self.learnable_molecule_encoder = Conditional_AGILE(self.args).to(self.args.device)
        for param in self.learnable_molecule_encoder.parameters():
            param.requires_grad = True
        if isinstance(self.regression_mlp, torch.nn.Sequential):
            for module in self.regression_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        elif hasattr(self.regression_mlp, "reset_parameters"):
            self.regression_mlp.reset_parameters()
        if isinstance(self.regression_mlp2, torch.nn.Sequential):
            for module in self.regression_mlp2:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        elif hasattr(self.regression_mlp2, "reset_parameters"):
            self.regression_mlp2.reset_parameters()
    
    
    def forward(self, molecules, ratios=None):
        ratios = _ensure_ratio_tensor(
            ratios,
            device=self.CLS_token.device,
            dtype=self.CLS_token.dtype,
        )
        batch_size = ratios.size(0)
        expected_graphs = batch_size * self.num_molecules
        if molecules.num_graphs != expected_graphs:
            raise ValueError(f"Expected {expected_graphs} molecule graphs, got {molecules.num_graphs}.")
        molecule_mask = _get_molecule_presence_mask(ratios, self.num_molecules)
        pooled_output = self.molecule_encoder(molecules, condition=None)
        pooled_output = pooled_output.reshape(batch_size, self.num_molecules, self.emb_dim)
        pooled_output = _masked_token_tensor(pooled_output, molecule_mask)
        
        ratios = ratios.clone()
        ratios[:, 0] = ratios[:, 1] / ratios[:, 0].clamp_min(1e-12)
        
        if self.pos_enc == "rand":
            pooled_output = pooled_output + self.rand_pos_emb.unsqueeze(0)
        RNA_token = self.RNA_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.emb_dim)
        cls_token = self.CLS_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.emb_dim)
        embeddings = torch.cat([cls_token, pooled_output, RNA_token], dim=1)
        cls_pos = ratios.sum(dim=1, keepdim=True)
        position_ids = torch.cat([cls_pos, ratios], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=ratios.device, dtype=torch.bool),
                molecule_mask,
                torch.ones(batch_size, 1, device=ratios.device, dtype=torch.bool),
            ],
            dim=1,
        )

        hidden_states = embeddings
        if self.pos_enc == "abs":
            hidden_states = hidden_states + _abs_positional_encoding(position_ids, self.emb_dim)
        hidden_states = self.emb_dropout(hidden_states)
        hidden_states = _masked_token_tensor(hidden_states, attention_mask)

        if self.pos_enc == "RoPE":
            rotary_position_ids = position_ids
        elif self.pos_enc == "RiPE":
            safe_position_ids = torch.where(
                attention_mask,
                position_ids.clamp_min(1e-12),
                torch.ones_like(position_ids),
            )
            rotary_position_ids = torch.log(safe_position_ids)
        else:
            rotary_position_ids = None

        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_position_ids, attention_mask=attention_mask)
            hidden_states = _masked_token_tensor(hidden_states, attention_mask)
            

        hidden_states = self.final_norm(hidden_states)
        hidden_states = _masked_token_tensor(hidden_states, attention_mask)
        cls_output = hidden_states[:, 0, :].unsqueeze(1).expand(batch_size, self.num_molecules, self.emb_dim)
        cls_output = _expanded_condition_from_token(cls_output, molecule_mask)
        pooled_output = self.learnable_molecule_encoder(molecules, condition=cls_output)
        pooled_output = pooled_output.reshape(batch_size, self.num_molecules, self.emb_dim)
        pooled_output = _masked_token_tensor(pooled_output, molecule_mask)
        regression_inputs = pooled_output.reshape(batch_size, -1)
        regression_output = self.regression_mlp(regression_inputs)
        other_outputs = hidden_states[:, 1:-1, :]
        other_outputs = _masked_token_tensor(other_outputs, molecule_mask)
        regression_output2 = self.regression_mlp2(other_outputs.reshape(batch_size, self.num_molecules * self.emb_dim))
        if self.args.consistency > 0:
            return regression_output, regression_output2, (other_outputs, molecule_mask), (pooled_output, molecule_mask)
        return regression_output, regression_output2, None, None
    
    
    def load_backbone_state_dict(self, state_dict):
        """Load pretrained weights into the AGILE backbone only."""
        self.molecule_encoder.load_my_state_dict(state_dict)
        if hasattr(self, "learnable_molecule_encoder"):
            self.learnable_molecule_encoder.load_my_state_dict(state_dict)

def _abs_positional_encoding(position_ids, dim):
    device = position_ids.device
    position_ids = position_ids.float().unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(position_ids.size(0), position_ids.size(1), dim, device=device)
    pe[..., 0::2] = torch.sin(position_ids * div_term)
    pe[..., 1::2] = torch.cos(position_ids * div_term)
    return pe


def _ensure_ratio_tensor(ratios, device, dtype):
    if ratios is None:
        raise ValueError("ratios must not be None.")
    if not torch.is_tensor(ratios):
        ratios = torch.tensor(ratios, device=device, dtype=dtype)
    else:
        ratios = ratios.to(device=device, dtype=dtype)
    if ratios.dim() == 3:
        ratios = ratios.squeeze(-1)
    return ratios


def _get_molecule_presence_mask(ratios, num_molecules):
    if ratios.size(1) < num_molecules:
        raise ValueError(
            f"Expected at least {num_molecules} ratio/scalar columns, got shape {tuple(ratios.shape)}."
        )
    return ratios[:, -num_molecules:] > 0


def _masked_token_tensor(token_tensor, token_mask):
    return token_tensor * token_mask.to(dtype=token_tensor.dtype).unsqueeze(-1)


def _expanded_condition_from_token(token_tensor, token_mask):
    batch_size, num_molecules, emb_dim = token_tensor.shape
    expanded = token_tensor.reshape(batch_size * num_molecules, emb_dim)
    expanded_mask = token_mask.reshape(batch_size * num_molecules, 1).to(dtype=token_tensor.dtype)
    return expanded * expanded_mask


class _InteractionSelfAttention(torch.nn.Module):
    def __init__(self, args, emb_dim, dropout, use_rotary):
        super(_InteractionSelfAttention, self).__init__()
        self.args = args
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // self.args.heads
        self.use_rotary = use_rotary
        self.qkv = torch.nn.Linear(emb_dim, emb_dim * 3)
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = torch.nn.Dropout(dropout)
        if use_rotary:
            self.rotary = RotaryEmbedding( self.head_dim, learnable_base=False)
        else:
            self.rotary = None

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.args.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.args.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.args.heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            assert position_ids != None
            cos, sin = self.rotary(position_ids, dtype=value.dtype)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if attention_mask is None and _FLASH_ATTN_AVAILABLE:
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
            dropout_p = self.attn_dropout.p if self.training else 0.0
            context = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=None, causal=False)
            context = context.contiguous().view(batch_size, seq_len, self.emb_dim)
            return self.out_proj(context)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim)
        return self.out_proj(context)


class _InteractionTransformerLayer(torch.nn.Module):
    def __init__(self, args, emb_dim, ffn_dim, dropout, use_rotary):
        super(_InteractionTransformerLayer, self).__init__()
        self.args = args
        self.attn_norm = torch.nn.LayerNorm(emb_dim)
        self.attn = _InteractionSelfAttention(args, emb_dim, dropout, use_rotary)
        self.ffn_norm = torch.nn.LayerNorm(emb_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, emb_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        attn_out = self.attn(self.attn_norm(hidden_states), position_ids, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        ffn_out = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + ffn_out
        return hidden_states
