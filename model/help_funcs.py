# import torch
# import torch.nn.functional as F
import paddle.nn.functional as F
# from torch import nn
import paddle
from paddle import nn
import paddlenlp

##  w未完成 ##

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2D(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias_attr=False),
                         nn.BatchNorm2D(in_channels),
                         nn.ReLU(),
                         nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_k = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(dim, inner_dim, bias_attr=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q = q.reshape((b, n, h, -1)).transpose((0, 2, 1, 3))
        k = k.reshape((b, m.shape[1], h, -1)).transpose((0, 2, 1, 3))
        v = v.reshape((b, m.shape[1], h, -1)).transpose((0, 2, 1, 3))

        dots = paddle.matmul(q, k, transpose_y=True) * self.scale
        mask_value = -65535

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = paddle.nn.functional.softmax(dots, axis=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = paddle.matmul(attn, v)
        out = out.transpose((0,2,1,3)).flatten(2)
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = map(lambda t: t.reshape((b, n, h, -1)).transpose((0, 2, 1, 3)), qkv)

        dots = paddle.matmul(q, k, transpose_y=True) * self.scale
        mask_value = -65535

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = paddle.nn.functional.softmax(dots, axis=-1)

        out = paddle.matmul(attn, v)
        out = out.transpose((0,2,1,3)).flatten(2)
        out = self.to_out(out)
        return out


class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

from scipy.io import savemat
def save_to_mat(x1, x2, fx1, fx2, cp, file_name):
    #Save to mat files
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        
        fx1_0_np = fx1[0].detach().cpu().numpy()
        fx2_0_np = fx2[0].detach().cpu().numpy()
        fx1_1_np = fx1[1].detach().cpu().numpy()
        fx2_1_np = fx2[1].detach().cpu().numpy()
        fx1_2_np = fx1[2].detach().cpu().numpy()
        fx2_2_np = fx2[2].detach().cpu().numpy()
        fx1_3_np = fx1[3].detach().cpu().numpy()
        fx2_3_np = fx2[3].detach().cpu().numpy()
        fx1_4_np = fx1[4].detach().cpu().numpy()
        fx2_4_np = fx2[4].detach().cpu().numpy()
        
        cp_np = cp[-1].detach().cpu().numpy()

        mdic = {'x1': x1_np, 'x2': x2_np, 
                'fx1_0': fx1_0_np, 'fx1_1': fx1_1_np, 'fx1_2': fx1_2_np, 'fx1_3': fx1_3_np, 'fx1_4': fx1_4_np,
                'fx2_0': fx2_0_np, 'fx2_1': fx2_1_np, 'fx2_2': fx2_2_np, 'fx2_3': fx2_3_np, 'fx2_4': fx2_4_np,
                "final_pred": cp_np}
                
        savemat("/media/lidan/ssd2/ChangeFormer/vis/mat/"+file_name+".mat", mdic)



