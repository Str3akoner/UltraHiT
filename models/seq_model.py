import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from functools import partial
import timm

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class IdentityModule(nn.Module):
    def forward(self, x):
        return x

class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]  

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v,
                                                     dropout_p=self.proj_drop_prob if self.training else 0.0,
                                                     is_causal=True)
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale 
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            attn = attn.masked_fill(~mask, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v 

        out = out.transpose(1, 2).reshape(B, T, C) 
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CausalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DiscreteActor(nn.Module):
    def __init__(self, in_features, num_actions, dropout, arch=None) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_actions),
        )

    def forward(self, x):
        return self.trunk(x)

class ClassificationModelSeq(nn.Module):
    def __init__(self,
                 arch: str,
                 num_actions: int,
                 dropout: float = 0.0,
                 action_type: str = 'discrete',
                 pretrain_backbbone: bool = True,  
                 max_frames: int = 5,
                 model_dim: int = 256,
                 num_blocks: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.1,
                 action_emb_dim: int = 128,
                 use_distancer: bool = False):
        super().__init__()
        assert action_type == 'discrete', "Current implementation is for discrete actions only"

        self.arch = arch
        self.num_actions = num_actions
        self.dropout = dropout
        self.max_frames = max_frames
        self.model_dim = model_dim
        self.use_distancer = use_distancer

        if 'resnet' in arch:
            base = tv_models.__dict__[arch](weights="IMAGENET1K_V2" if pretrain_backbbone else None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        elif 'convnext' in arch:
            base = tv_models.__dict__[arch](weights="IMAGENET1K_V1" if pretrain_backbbone else None)
            in_features = base.classifier[2].in_features
            base.classifier = IdentityModule()
        elif 'deit' in arch:
            deit_name = 'deit_tiny_patch16_224' if arch in (
                'deit_tiny', 'deit_tiny_224', 'deit_tiny_patch16'
            ) else arch
            base = timm.create_model(
                deit_name,
                pretrained=pretrain_backbbone,
                num_classes=0,   
                global_pool='avg' 
            )
            in_features = base.num_features 
        else:
            raise NotImplementedError(f"Unsupported arch: {arch}")
        self.backbone = base
        self.backbone_out = in_features

        self.img_proj = nn.Linear(self.backbone_out, model_dim)   
        self.act_embed = nn.Embedding(13, action_emb_dim)  
        self.start_act = nn.Parameter(torch.zeros(action_emb_dim))  
        self.act_proj = nn.Sequential(                           
            nn.Linear(action_emb_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, model_dim * 2))
        trunc_normal_(self.pos_emb, std=0.02)

        blocks = []
        dim = model_dim * 2
        for _ in range(num_blocks):
            blocks.append(CausalBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                      qkv_bias=True, drop=drop, attn_drop=drop))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.LayerNorm(dim)

        self.actors = DiscreteActor(dim, num_actions, dropout, arch)
        self.distancers = DiscreteActor(dim, 6, dropout, arch) if use_distancer else None


    def forward(self, img_seq: torch.Tensor, past_actions: torch.Tensor):
        """
        img_seq: [B, T, C, H, W]
        past_actions: [B, T-1]
        """
        B, T, C, H, W = img_seq.shape
        assert T <= self.max_frames, f"T={T} exceed max_frames={self.max_frames}"
        assert past_actions.shape[1] == T - 1, "past_actions should have length T-1"

        x = img_seq.view(B * T, C, H, W)

        feats = self.backbone(x)      
        feats = self.img_proj(feats)  
        feats = feats.view(B, T, -1)  

        a = self.act_embed(past_actions)            
        start = self.start_act.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  
        a = torch.cat([start, a], dim=1)           
        a = self.act_proj(a)                       

        h = torch.cat([feats, a], dim=-1)         

        h = h + self.pos_emb[:, :T, :]

        h = self.blocks(h)                        
        h = self.norm_out(h)

        fused = h[:, -1, :]                      
        policy = self.actors(fused)               
        out = {'policy': policy, 'fused': fused}
        if self.distancers is not None:
            out['dist'] = self.distancers(fused)    
        return out

    def get_actor_output(self, features):
        return self.actors(features)

    def get_distancer_output(self, features):
        assert self.distancers is not None, "distancer head disabled"
        return self.distancers(features)


