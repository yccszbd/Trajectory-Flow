import math

import numpy as np
import torch
import torch.nn.functional as F
from tools.utils import extract, gen_coefficients
from torch import nn

from models.embedder import get_embedder


class CAPUDFNetwork(nn.Module):
    def __init__(
        self,
        time_sum,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        film_in=(4),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__()
        self.time_sum = time_sum
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        # [3,256,256,256,256,256,256,256,256,1]
        # self.Film_layers = [4, 8]
        self.embed_fn_fine = None
        # self.pe_t = TimeStepEncoding()
        input_dim = dims[0]
        if multires > 0:
            embed_fn, input_dim = get_embedder(multires, input_dims=d_in)
            # embed_fn是特征转高频模块,multires表示转的高频模块的频率上线,最高sin4x则multires=4,输出的特征为[x,sinx,cosx,sin2x,cos2x,sin3x,cos3x,sin4x,cos4x],维度一共为3*(1+2*multires)
            self.embed_fn_fine = embed_fn
            dims[0] = input_dim

        # 可学习的正余弦时间编码,长度设置为默认16
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim, self.time_sum)
        fourier_dim = learned_sinusoidal_dim + 1
        # 对输入层的时间调制
        self.time_mlp1 = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, 2 * input_dim), nn.GELU(), nn.Linear(2 * input_dim, 2 * input_dim)
        )
        # 对线性层的的时间调制
        self.time_mlp2 = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, 2 * d_hidden), nn.GELU(), nn.Linear(2 * d_hidden, 2 * d_hidden)
        )

        self.num_layers = len(dims)
        # len(dims) = 10
        self.skip_in = skip_in
        self.film_in = film_in
        self.scale = scale
        # dims[0]=3,这设置9层
        for l in range(self.num_layers - 1):  # noqa: E741
            if l + 1 in self.skip_in:
                # 如果下一层要进行跳跃连接则在该层的输出要减少一点以保证维度一致
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        # self.activation = nn.Softplus(beta=100)
        self.activation = nn.ReLU()

    def forward(self, query, time):
        query = query * self.scale
        if self.embed_fn_fine is not None:
            query = self.embed_fn_fine(query)
        q_feature = query
        time_emb_1 = self.time_mlp1(time)
        scale_1, shift_1 = torch.chunk(time_emb_1, 2, dim=-1)
        q_feature = q_feature * (1 + scale_1) + shift_1
        time_emb_2 = self.time_mlp2(time)
        scale_2, shift_2 = torch.chunk(time_emb_2, 2, dim=-1)

        for l in range(self.num_layers - 1):  # noqa: E741
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                q_feature = torch.cat([q_feature, query], -1) / np.sqrt(2)
            if l in self.film_in:
                q_feature = q_feature * (1 + scale_2) + shift_2
            # 保证整体方差一致
            q_feature = lin(q_feature)

            if l < self.num_layers - 2:
                q_feature = self.activation(q_feature)

        res = torch.abs(q_feature)
        return res / self.scale

    def res_udf(self, query, time):
        return self.forward(query, time)

    def udf_hidden_appearance(self, query, time):
        return self.forward(query, time)

    def gradient(self, x, time):
        x.requires_grad_(True)
        y = self.res_udf(x, time)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return gradients

    def predict(self, x, time_sum, metric="increased"):
        """
        进行多步移动
        """
        current_points = x
        time = time_sum - 1  # 假设我们从时间步10开始
        alphas = gen_coefficients(time_sum, schedule=metric).to(device=current_points.device)
        # 在evaluate阶段是的
        while time >= 0:
            # 明确require_grad以确保可以在后面求gradient
            current_points.requires_grad = True
            time_list = torch.full(
                (*current_points.shape[:-1], 1),
                time,
                dtype=torch.int64,
                device=current_points.device,
            )
            gradients_sample = self.gradient(current_points, time_list)  # 5000x3
            udf_sample = self.res_udf(current_points, time_list)  # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=-1)  # 5000x3
            coef = extract(alphas, time_list)  # 5000x1
            p_next = current_points - grad_norm * udf_sample * coef
            # 更新当前点
            current_points = p_next.detach()
            time -= 1

        udf = torch.norm(x - current_points, dim=-1, keepdim=True)
        gradient = (x - current_points) / udf
        return udf, gradient, current_points


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int, timesum):
        super().__init__()
        assert dim % 2 == 0
        self.include_input = True
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(self.half_dim))  # [D/2]
        self.timesum = timesum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 做归一化不贴边
        x = (2 * (x + 0.5) / self.timesum) - 1
        # 广播: weights -> [1,1,D/2]
        w = self.weights.view(*([1] * (x.dim() - 1)), self.half_dim)

        # 相位 & 特征: [B,N,D/2] -> concat sin/cos -> [B,N,dim]
        freqs = x * w * (2 * math.pi)
        feat = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # [B,N,dim]

        return torch.cat((x, feat), dim=-1)
