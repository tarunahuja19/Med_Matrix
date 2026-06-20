import torch
import torch.nn as nn
import numpy as np

class RealValuedComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=False)
        self.fc_imag = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.fc_real(xr) - self.fc_imag(xi)
        out_imag = self.fc_real(xi) + self.fc_imag(xr)
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        return out_real, out_imag

class RealValuedComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_imag = nn.LayerNorm(normalized_shape, eps=eps)
        
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ln_real(xr), self.ln_imag(xi)

class RealValuedSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.log_re_a = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.im_a = nn.Parameter(torch.pi * torch.arange(d_state).unsqueeze(0).repeat(d_model, 1))
        
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        
        self.log_dt = nn.Parameter(torch.log(torch.exp(torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))))
        
        self.d_real = nn.Parameter(torch.ones(d_model))
        self.d_imag = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = xr.shape
        device = xr.device
        
        a_real = -torch.exp(self.log_re_a)
        a_imag = self.im_a
        dt = torch.exp(self.log_dt)
        
        dt_a_real = dt.unsqueeze(1) * a_real
        dt_a_imag = dt.unsqueeze(1) * a_imag
        
        n_real = 1.0 + 0.5 * dt_a_real
        n_imag = 0.5 * dt_a_imag
        d_real = 1.0 - 0.5 * dt_a_real
        d_imag = -0.5 * dt_a_imag
        
        denom = d_real**2 + d_imag**2
        dA_real = (n_real * d_real + n_imag * d_imag) / denom
        dA_imag = (n_imag * d_real - n_real * d_imag) / denom
        
        m_real = dt.unsqueeze(1) * self.b_real
        m_imag = dt.unsqueeze(1) * self.b_imag
        dB_real = (m_real * d_real + m_imag * d_imag) / denom
        dB_imag = (m_imag * d_real - m_real * d_imag) / denom
        
        h_real = torch.zeros(B, D, self.d_state, dtype=torch.float32, device=device)
        h_imag = torch.zeros(B, D, self.d_state, dtype=torch.float32, device=device)
        
        y_real = torch.zeros(B, S, D, dtype=torch.float32, device=device)
        y_imag = torch.zeros(B, S, D, dtype=torch.float32, device=device)
        
        dA_real_uns = dA_real.unsqueeze(0)
        dA_imag_uns = dA_imag.unsqueeze(0)
        dB_real_uns = dB_real.unsqueeze(0)
        dB_imag_uns = dB_imag.unsqueeze(0)
        
        c_real_uns = self.c_real.unsqueeze(0)
        c_imag_uns = self.c_imag.unsqueeze(0)
        d_real_uns = self.d_real.unsqueeze(0)
        d_imag_uns = self.d_imag.unsqueeze(0)
        
        for k in range(S):
            xr_k = xr[:, k, :].unsqueeze(2)
            xi_k = xi[:, k, :].unsqueeze(2)
            
            trans_real = dA_real_uns * h_real - dA_imag_uns * h_imag
            trans_imag = dA_real_uns * h_imag + dA_imag_uns * h_real
            
            input_real = dB_real_uns * xr_k - dB_imag_uns * xi_k
            input_imag = dB_real_uns * xi_k + dB_imag_uns * xr_k
            
            h_real = trans_real + input_real
            h_imag = trans_imag + input_imag
            
            ch_real = torch.sum(c_real_uns * h_real - c_imag_uns * h_imag, dim=2)
            ch_imag = torch.sum(c_real_uns * h_imag + c_imag_uns * h_real, dim=2)
            
            dx_real = d_real_uns * xr[:, k, :] - d_imag_uns * xi[:, k, :]
            dx_imag = d_real_uns * xi[:, k, :] + d_imag_uns * xr[:, k, :]
            
            y_real[:, k, :] = ch_real + dx_real
            y_imag[:, k, :] = ch_imag + dx_imag
            
        return y_real, y_imag

class RealValuedSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.ln = RealValuedComplexLayerNorm(d_model)
        self.proj_inner = RealValuedComplexLinear(d_model, d_model * expand, bias=True)
        self.ssm = RealValuedSSM(d_model * expand, d_state=d_state)
        self.proj_out = RealValuedComplexLinear(d_model * expand, d_model, bias=True)
        
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res_r, res_i = xr, xi
        ln_r, ln_i = self.ln(xr, xi)
        proj_r, proj_i = self.proj_inner(ln_r, ln_i)
        ssm_r, ssm_i = self.ssm(proj_r, proj_i)
        out_r, out_i = self.proj_out(ssm_r, ssm_i)
        return res_r + out_r, res_i + out_i

class RealValuedKSpaceS4Classifier(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 16, n_layers: int = 2, num_classes: int = 11, input_dim: int = 16384):
        super().__init__()
        self.encoder = RealValuedComplexLinear(input_dim, d_model, bias=True)
        self.blocks = nn.ModuleList([
            RealValuedSSMBlock(d_model, d_state=d_state, expand=2)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, xr: torch.Tensor, xi: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        B, S, C, H, W = xr.shape
        xr_flat = xr.reshape(B, S, C * H * W)
        xi_flat = xi.reshape(B, S, C * H * W)
        
        u_real, u_imag = self.encoder(xr_flat, xi_flat)
        for block in self.blocks:
            u_real, u_imag = block(u_real, u_imag)
            
        if return_sequence:
            return torch.sqrt(u_real**2 + u_imag**2)
            
        up_real = torch.mean(u_real, dim=1)
        up_imag = torch.mean(u_imag, dim=1)
        mag = torch.sqrt(up_real**2 + up_imag**2)
        return self.head(mag)

class RealValuedIFFT2(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        m = torch.arange(size).unsqueeze(1)
        n = torch.arange(size).unsqueeze(0)
        angle = 2 * torch.pi * m * n / size
        self.register_buffer('W_R', torch.cos(angle) / size)
        self.register_buffer('W_I', torch.sin(angle) / size)
        
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wr_xr = torch.matmul(self.W_R, xr)
        wi_xi = torch.matmul(self.W_I, xi)
        wr_xi = torch.matmul(self.W_R, xi)
        wi_xr = torch.matmul(self.W_I, xr)
        
        a_real = wr_xr - wi_xi
        a_imag = wr_xi + wi_xr
        
        W_R_T = self.W_R.t()
        W_I_T = self.W_I.t()
        
        y_real = torch.matmul(a_real, W_R_T) - torch.matmul(a_imag, W_I_T)
        y_imag = torch.matmul(a_real, W_I_T) + torch.matmul(a_imag, W_R_T)
        
        return y_real, y_imag

class RealValuedVolumeCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 11, resolution: int = 128):
        super().__init__()
        self.recon = RealValuedIFFT2(resolution)
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.head = nn.Linear(128, num_classes)
        self.resolution = resolution
        
    def reconstruct_kspace(self, kr: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        shift_val = self.resolution // 2
        kr_shifted = torch.roll(kr, shifts=(shift_val, shift_val), dims=(-2, -1))
        ki_shifted = torch.roll(ki, shifts=(shift_val, shift_val), dims=(-2, -1))
        
        img_r, img_i = self.recon(kr_shifted, ki_shifted)
        
        coil_r = torch.roll(img_r, shifts=(shift_val, shift_val), dims=(-2, -1))
        coil_i = torch.roll(img_i, shifts=(shift_val, shift_val), dims=(-2, -1))
        
        combined = torch.sqrt(torch.sum(coil_r**2 + coil_i**2, dim=2))
        return combined.unsqueeze(1)
        
    def forward(self, kr: torch.Tensor, ki: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        mag = self.reconstruct_kspace(kr, ki)
        B, _, S, H, W = mag.shape
        
        x_feat = self.relu1(self.bn1(self.conv1(mag)))
        x_feat = self.relu2(self.bn2(self.conv2(x_feat)))
        x_feat = self.relu3(self.bn3(self.conv3(x_feat)))
        x_feat = self.relu4(self.bn4(self.conv4(x_feat)))
        
        if return_sequence:
            # Global spatial average: [B, C, S_down, H', W'] → [B, C, S_down]
            x_spat_pool = torch.mean(x_feat, dim=[3, 4])
            # Interpolate sequence length back to S (original temporal length)
            x_seq = nn.functional.interpolate(x_spat_pool, size=S, mode='linear', align_corners=False)
            # Transpose to [B, S, C]
            return x_seq.transpose(1, 2)
            
        # Manual global average pool (ONNX-compatible: avoid AdaptiveAvgPool3d)
        x_pool = torch.mean(x_feat, dim=[2, 3, 4])  # [B, 128]
        return self.head(x_pool)

class FusedS4CNNClassifierONNX(nn.Module):
    def __init__(self, d_model_s4: int = 128, d_state_s4: int = 16, n_layers_s4: int = 2, d_model_cnn: int = 128, num_classes: int = 11, input_dim_s4: int = 16384, d_attn: int = 128, resolution: int = 128):
        super().__init__()
        self.s4_branch = RealValuedKSpaceS4Classifier(
            d_model=d_model_s4,
            d_state=d_state_s4,
            n_layers=n_layers_s4,
            num_classes=num_classes,
            input_dim=input_dim_s4
        )
        self.cnn_branch = RealValuedVolumeCNNClassifier(
            num_classes=num_classes,
            resolution=resolution
        )
        
        self.q_proj = nn.Linear(d_model_cnn, d_attn)
        self.k_proj = nn.Linear(d_model_s4, d_attn)
        self.v_proj = nn.Linear(d_model_s4, d_attn)
        self.res_proj = nn.Linear(d_model_cnn, d_attn)
        
        self.head = nn.Linear(d_attn, num_classes)
        self.d_attn = d_attn
        
    def forward(self, x_real_imag: torch.Tensor) -> torch.Tensor:
        xr = x_real_imag[..., 0]
        xi = x_real_imag[..., 1]
        
        z_s4 = self.s4_branch(xr, xi, return_sequence=True)
        z_cnn = self.cnn_branch(xr, xi, return_sequence=True)
        
        Q = self.q_proj(z_cnn)
        K = self.k_proj(z_s4)
        V = self.v_proj(z_s4)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_attn ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        z_attn = torch.matmul(attn_weights, V)
        
        z_fused = z_attn + self.res_proj(z_cnn)
        f_pool = torch.mean(z_fused, dim=1)
        return self.head(f_pool)
