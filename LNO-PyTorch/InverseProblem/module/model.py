import torch
import math


def Attention_Linear_GNOT(q, k, v):
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    k_sum = k.sum(dim=-2, keepdim=True)
    inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)
    r = q + (q @ (k.transpose(-2, -1) @ v)) * inv
    return r


def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r


ACTIVATION = {"Sigmoid": torch.nn.Sigmoid(),
              "Tanh": torch.nn.Tanh(),
              "ReLU": torch.nn.ReLU(),
              "LeakyReLU": torch.nn.LeakyReLU(0.1),
              "ELU": torch.nn.ELU(),
              "GELU": torch.nn.GELU()
              }


ATTENTION = {"HNA": Attention_Linear_GNOT,
             "Attention_Vanilla": Attention_Vanilla
            }


class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.n_layer = n_layer
            self.act = act
            self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
            self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
            
        def forward(self, x):
            r = self.act(self.input(x))
            for i in range(0, self.n_layer):
                r = r + self.act(self.hidden[i](r))
            r = self.output(r)
            return r
        

class GNOT(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_dim, n_head, attn):
            super().__init__()
            self.n_dim = n_dim
            self.n_head = n_head
            self.key = torch.nn.Linear(self.n_dim, self.n_dim)
            self.query = torch.nn.Linear(self.n_dim, self.n_dim)
            self.value = torch.nn.Linear(self.n_dim, self.n_dim)
            # regularization
            self.drop = torch.nn.Dropout(0.0)
            # attention type
            self.attn = attn
            # output projection
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, x):
            B, N, D = x.size()
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q = self.query(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.key(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.value(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            
            r = self.attn(q, k, v)
            r = self.drop(r)
            r = r.permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r

    class CrossAttention(torch.nn.Module):
        def __init__(self, n_dim, n_head, attn):
            super().__init__()
            self.n_dim = n_dim
            self.n_head = n_head
            self.key = torch.nn.Linear(self.n_dim, self.n_dim)
            self.query = torch.nn.Linear(self.n_dim, self.n_dim)
            self.value = torch.nn.Linear(self.n_dim, self.n_dim)
            # regularization
            self.drop = torch.nn.Dropout(0.0)
            # attention type
            self.attn = attn
            # output projection
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)

        def forward(self, x, y):
            B, N1, D = x.size()
            _, N2, _ = y.size()
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q = self.query(x).view(B, N1, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.key(y).view(B, N2, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.value(y).view(B, N2, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            
            r = self.attn(q, k, v)
            r = self.drop(r)
            r = r.permute(0, 2, 1, 3).contiguous().view(B, N1, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_dim, n_head, n_expert, attn, act, x_dim):
            super().__init__()
            self.n_dim = n_dim
            self.n_head = n_head
            self.n_expert = n_expert
            self.attn = attn
            self.act = act
            self.x_dim = x_dim
            
            self.self_attn = GNOT.SelfAttention(self.n_dim, self.n_head, self.attn)
            self.cross_attn = GNOT.CrossAttention(self.n_dim, self.n_head, self.attn)
            
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.ln3 = torch.nn.LayerNorm(self.n_dim)
            self.ln4 = torch.nn.LayerNorm(self.n_dim)
            self.ln5 = torch.nn.LayerNorm(self.n_dim)

            self.drop1 = torch.nn.Dropout(0.0)
            self.drop2 = torch.nn.Dropout(0.0)
            
            self.moe_mlp1 = torch.nn.ModuleList([torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            ) for _ in range(self.n_expert)])
            
            self.moe_mlp2 = torch.nn.ModuleList([torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            ) for _ in range(self.n_expert)])
            
            self.gatenet = torch.nn.Sequential(
                torch.nn.Linear(self.x_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_expert)
            )

        def forward(self, x, y, pos):        
            gate_score = torch.nn.functional.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
            r = x + self.drop1(self.cross_attn(self.ln1(x), self.ln2(y)))
            r_moe1 = torch.stack([self.moe_mlp1[i](r) for i in range(self.n_expert)], dim=-1)
            r_moe1 = torch.sum(gate_score * r_moe1, dim=-1)
            r = r + self.ln3(r_moe1)
            r = r + self.drop2(self.self_attn(self.ln4(r)))
            r_moe2 = torch.stack([self.moe_mlp2[i](r) for i in range(self.n_expert)], dim=-1)
            r_moe2 = torch.sum(gate_score * r_moe2, dim=-1)
            r = r + self.ln5(r_moe2)
            return r
        
    def __init__(self, n_block, n_dim, n_head, n_layer, n_expert, x_dim, y_dim, f_dim, attn, act):
        super().__init__()
        self.n_block = n_block
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_expert = n_expert

        self.attn = ATTENTION[attn]
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.f_dim = f_dim
        
        self.trunk_mlp = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_mlp = MLP(self.x_dim+self.y_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, y_dim, self.n_layer, self.act)

        self.attn_blocks = torch.nn.Sequential(*[GNOT.AttentionBlock(self.n_dim, self.n_head, self.n_expert, self.attn, self.act, self.x_dim) for _ in range(self.n_block)])


    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    def forward(self, x, y):
        pos = x[:, :, 0:self.x_dim]
        x = self.trunk_mlp(x)
        y = self.branch_mlp(y)
        for block in self.attn_blocks:
            x = block(x, y, pos)
        r = self.out_mlp(x)
        return r


class DeepONet(torch.nn.Module):
    class Branch(torch.nn.Module):
        def __init__(self, n, p):
            super().__init__()
            self.n = n
            self.p = p
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(self.p, nhead=4, dim_feedforward=self.p*2, batch_first=True), 4)
            self.feature = torch.nn.Parameter(torch.zeros(1, 1, self.p))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n, int(self.p/2)),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(int(self.p/2), self.p),
                )
            
        def forward(self, x):
            feature = self.feature.expand(x.shape[0], -1, -1)
            x = self.mlp(x)
            x = torch.cat((feature, x), dim=1)
            r = torch.reshape(self.encoder(x)[:,0,:].contiguous(), (x.shape[0], 1, self.p)).contiguous()
            return r
    
    class Trunk(torch.nn.Module):
        def __init__(self, n, p):
            super().__init__()
            self.n = n
            self.p = p
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n, int(self.p/2)),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(int(self.p/2), int(self.p/2)),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(int(self.p/2), self.p),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.p, self.p),
                torch.nn.LeakyReLU()
                )

        def forward(self, x):
            r = self.mlp(x)
            return r
    
    class Dot(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.p, self.p),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.p, 1),
                )

        def forward(self, t, b):
            r = t * b
            r = self.mlp(r)
            return r
    
    def __init__(self, x_dim, y_dim, p):
        super().__init__()
        self.p = p
        self.n_branch = x_dim + y_dim 
        self.n_trunk = x_dim
        self.branch = DeepONet.Branch(self.n_branch, self.p)
        self.trunk = DeepONet.Trunk(self.n_trunk, self.p)
        self.dot = DeepONet.Dot(self.p)

    def forward(self, x, ob):
        t = self.trunk(x)
        b = self.branch(ob)
        r = self.dot(t, b)
        return r
    

class LNO(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class CrossAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, y, x):
            B, N, D = x.size()
            _, M, _ = y.size()
            q = self.Wq(y).view(B, M, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, M, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            self.self_attn = LNO.SelfAttention(self.n_mode, self.n_dim, self.n_head, self.attn)
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y
        
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, attn, act):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer

        self.attn = ATTENTION[attn]
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        self.y2_dim = y2_dim
        
        self.trunk_mlp = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_mlp = MLP(self.x_dim + self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)

        self.mode_mlp = torch.nn.Linear(self.n_dim, self.n_mode)
        self.encode_mlp = torch.nn.Linear(self.n_dim, self.n_dim)
        self.decode_mlp = torch.nn.Linear(self.n_dim, self.n_dim)
        
        self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
        self.ln = torch.nn.LayerNorm(self.n_dim)

        self.attn_blocks = torch.nn.Sequential(*[LNO.AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.attn, self.act) for _ in range(0, self.n_block)])
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, query, y):
        x = self.trunk_mlp(y[...,:self.x_dim])
        query = self.trunk_mlp(query)
                
        score_encode = torch.softmax(self.mode_mlp(x), dim=1)
        score_decode = torch.softmax(self.mode_mlp(query), dim=-1)
        
        y = self.branch_mlp(y)
        v = self.Wv(self.ln(y))

        z = torch.einsum("bij,bic->bjc", score_encode, v)
        z = self.encode_mlp(z)
            
        for block in self.attn_blocks:
            z = block(z)
        
        z = self.decode_mlp(z)
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        
        r = self.out_mlp(r)

        return r
