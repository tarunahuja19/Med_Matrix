import torch
import torch.nn as nn
import numpy as np

class ComplexLinear(nn.Module):
    """
    Complex-valued Linear Layer.
    Applies W * X + b for complex weights W, bias b, and input X.
    Uses real-valued PyTorch Linear layers under the hood for stability.
    """
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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is complex64
        x_real = x.real
        x_imag = x.imag
        
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
            
        return torch.complex(out_real, out_imag)

class ComplexLayerNorm(nn.Module):
    """
    Complex-valued Layer Normalization.
    Applies LayerNorm separately to the real and imaginary parts.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_imag = nn.LayerNorm(normalized_shape, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = self.ln_real(x.real)
        x_imag = self.ln_imag(x.imag)
        return torch.complex(x_real, x_imag)

class ComplexSSM(nn.Module):
    """
    Complex-valued Diagonal Structured State Space (S4D) Layer.
    Discretizes a continuous diagonal SSM using Bilinear transform:
      h_k = dA * h_{k-1} + dB * x_k
      y_k = sum(c * h_k) + d * x_k
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Learnable continuous-time transition parameters
        # Re(A) must be negative to ensure stability, so we learn log(-Re(A))
        # Initialized similar to HiPPO matrix diagonal: Re(A) = -0.5, Im(A) = n * pi
        self.log_re_a = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.im_a = nn.Parameter(torch.pi * torch.arange(d_state).unsqueeze(0).repeat(d_model, 1))
        
        # Learnable complex B parameter
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        
        # Learnable complex C parameter
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        
        # Learnable step size delta, initialized between 0.001 and 0.1
        self.log_dt = nn.Parameter(torch.log(torch.exp(torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))))
        
        # Feedforward term D
        self.d_real = nn.Parameter(torch.ones(d_model))
        self.d_imag = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, S, D] (complex64)
        B, S, D = x.shape
        device = x.device
        
        # Reconstruct continuous parameters
        a_real = -torch.exp(self.log_re_a) # negative real part guarantees stability
        a_imag = self.im_a
        a = torch.complex(a_real, a_imag) # [D, N]
        
        b = torch.complex(self.b_real, self.b_imag) # [D, N]
        c = torch.complex(self.c_real, self.c_imag) # [D, N]
        dt = torch.exp(self.log_dt) # [D]
        d = torch.complex(self.d_real, self.d_imag) # [D]
        
        # Bilinear discretization:
        # dA = (1 + dt * A / 2) / (1 - dt * A / 2)
        dt_a = dt.unsqueeze(1) * a
        dA = (1.0 + dt_a * 0.5) / (1.0 - dt_a * 0.5) # [D, N]
        
        # dB = dt * B / (1 - dt * A / 2)
        dB = (dt.unsqueeze(1) * b) / (1.0 - dt_a * 0.5) # [D, N]
        
        # Recurrent scan loop
        # State h shape: [B, D, N]
        h = torch.zeros(B, D, self.d_state, dtype=torch.complex64, device=device)
        y = torch.zeros(B, S, D, dtype=torch.complex64, device=device)
        
        for k in range(S):
            x_k = x[:, k, :] # [B, D]
            # State update: h = dA * h + dB * x_k
            h = dA.unsqueeze(0) * h + dB.unsqueeze(0) * x_k.unsqueeze(2)
            # Output update: y_k = sum(c * h_k) + d * x_k
            y_k = torch.sum(c.unsqueeze(0) * h, dim=2) + d.unsqueeze(0) * x_k
            y[:, k, :] = y_k
            
        return y

class ComplexSSMBlock(nn.Module):
    """
    Complex-valued SSM Layer block with residual connection.
    Implements: LN -> ProjIn -> ComplexSSM -> ProjOut + Residual
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.ln = ComplexLayerNorm(d_model)
        self.proj_inner = ComplexLinear(d_model, d_model * expand, bias=True)
        self.ssm = ComplexSSM(d_model * expand, d_state=d_state)
        self.proj_out = ComplexLinear(d_model * expand, d_model, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, D] (complex64)
        residual = x
        x_norm = self.ln(x)
        x_proj = self.proj_inner(x_norm)
        x_ssm = self.ssm(x_proj)
        x_out = self.proj_out(x_ssm)
        return residual + x_out

class KSpaceS4Encoder(nn.Module):
    """
    Parameter-efficient 2D Convolutional Encoder for Complex K-space Slices.
    Maps [B, S, C, H, W] complex tensor to [B, S, d_model] complex representation.
    """
    def __init__(self, coils: int, d_model: int):
        super().__init__()
        # Input has 2 * coils channels (real and imag parts stacked)
        self.conv1 = nn.Conv2d(2 * coils, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Project real representation back to complex
        self.proj_real = nn.Linear(128, d_model)
        self.proj_imag = nn.Linear(128, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, C, H, W] (complex64)
        B, S, C, H, W = x.shape
        # Stack real and imaginary parts along the channel dimension
        # Reshape to [B * S, C, H, W]
        x_real = x.real.reshape(B * S, C, H, W)
        x_imag = x.imag.reshape(B * S, C, H, W)
        x_in = torch.cat([x_real, x_imag], dim=1) # [B * S, 2 * C, H, W]
        
        feat = self.relu1(self.bn1(self.conv1(x_in)))
        feat = self.relu2(self.bn2(self.conv2(feat)))
        feat = self.relu3(self.bn3(self.conv3(feat)))
        feat = self.relu4(self.bn4(self.conv4(feat)))
        
        feat_pooled = self.pool(feat).view(B * S, 128)
        
        out_real = self.proj_real(feat_pooled).view(B, S, -1)
        out_imag = self.proj_imag(feat_pooled).view(B, S, -1)
        
        return torch.complex(out_real, out_imag)

class KSpaceS4Classifier(nn.Module):
    """
    Classifier for raw patient K-space volumes.
    Accepts [B, S, C, H, W] complex inputs, projects them via a Conv2D encoder,
    passes them through sequential Complex SSM blocks, pools them, and predicts logits.
    """
    def __init__(self, d_model: int = 128, d_state: int = 16, n_layers: int = 2, num_classes: int = 11, input_dim: int = 16384, coils: int = 16):
        super().__init__()
        self.encoder = KSpaceS4Encoder(coils=coils, d_model=d_model)
        self.blocks = nn.ModuleList([
            ComplexSSMBlock(d_model, d_state=d_state, expand=2)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor, return_features: bool = False, return_sequence: bool = False) -> torch.Tensor:
        # Input shape: [B, S, C, H, W] (complex64)
        
        # Project to latent dimension via Conv2D encoder
        x_enc = self.encoder(x) # [B, S, d_model] (complex64)
        
        # Sequence processing
        x_out = x_enc
        for block in self.blocks:
            x_out = block(x_out) # [B, S, d_model]
            
        if return_sequence:
            return torch.abs(x_out) # [B, S, d_model]
            
        # Global average pooling over slices
        x_pool = torch.mean(x_out, dim=1) # [B, d_model]
        
        # Phase-invariant magnitude representation
        x_mag = torch.abs(x_pool) # [B, d_model]
        
        if return_features:
            return x_mag
            
        # Classification logits
        logits = self.head(x_mag) # [B, num_classes]
        return logits
