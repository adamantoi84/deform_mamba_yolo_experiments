"""
DeformSS2D — Deformable Selective Scan 2D Module
==================================================

Subclass dari SS2D di Mamba-YOLO yang mengganti scanning tetap (CrossScan 4 arah)
dengan deformable scanning ala DefMamba (Liu et al., CVPR 2025).

Tiga mode konfigurabel via `deform_mode`:
  - "k3_replace"      : K=3 (1 forward + 1 backward + 1 deformable) — ikuti DefMamba persis
  - "k5_augment"      : K=5 (4 SS2D + 1 deformable) — paling kaya
  - "k4_replace_all"  : K=4 deformable semua arah — paling agresif

Implementasi pure-PyTorch (tanpa kernel CUDA baru) untuk kemudahan debug.
Tetap memakai kernel selective_scan CUDA yang sudah ada di Mamba-YOLO.

Penempatan: drop file ini ke ultralytics/nn/modules/deform_ss2d.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

# Import dari Mamba-YOLO
from .mamba_yolo import SS2D, VSSBlock, XSSBlock
from .common_utils_mbyolo import SelectiveScanCore


# ============================================================================
#  Offset Network — meniru DefMamba ConvOffset
# ============================================================================

class ConvOffset(nn.Module):
    """
    Memprediksi offset (Δy, Δx) + path index (Δp) dari feature map.

    Input:  (B, C, H, W)
    Output: (B, 3, H, W)  — channel 0,1 = Δy, Δx; channel 2 = Δp (path)
    """
    def __init__(self, in_ch: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2 if kernel_size > 1 else 0
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.norm = nn.LayerNorm(in_ch)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                padding=pad, groups=in_ch)
        # Channel attention sederhana (squeeze-excitation)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 4, 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(in_ch // 4, 4), in_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        # Output 3 channel: Δy, Δx, Δp
        self.out = nn.Conv2d(in_ch, 3, kernel_size=1)
        # Init offset kecil supaya awal training mirip SS2D
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.proj(x)
        # LayerNorm di channel: pindah ke (B, H, W, C)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.act(y)
        y = self.dwconv(y)
        y = y * self.ca(y)
        return self.out(y)  # (B, 3, H, W)


# ============================================================================
#  Deformable Scan — versi pure-PyTorch
# ============================================================================

class DeformableScan(nn.Module):
    """
    Menggantikan CrossScan untuk satu arah deformable.

    Inti operasi (mengikuti DefMamba):
      1. Prediksi offset (Δy, Δx) + path index Δp dari fitur input
      2. Sample fitur di posisi adaptif p̂ = p_ref + Δp via bilinear interpolation
      3. Sort token berdasarkan posisi path adaptif → urutan deformable
      4. Return token urut + indeks untuk un-sort nanti

    Input:  x (B, C, H, W)
    Output: x_deformed (B, C, N), indices (B, N)
            di mana N = H*W
    """
    def __init__(self, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv_offset = ConvOffset(embed_dim, kernel_size)

    @torch.no_grad()
    def _ref_grid(self, H: int, W: int, B: int, dtype, device):
        """Reference grid normalized [-1, 1] untuk grid_sample."""
        ry, rx = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ry, rx), -1)  # (H, W, 2)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        return ref[None].expand(B, -1, -1, -1)  # (B, H, W, 2)

    @torch.no_grad()
    def _path_ref(self, N: int, B: int, dtype, device):
        """Path reference 1D normalized [-1, 1]."""
        ref = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device)
        ref = ref.div(N - 1.0).mul(2.0).sub(1.0)
        return ref[None].expand(B, -1)  # (B, N)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W
        dtype, device = x.dtype, x.device

        # 1. Prediksi offset (Δy, Δx, Δp)
        offset = self.conv_offset(x)  # (B, 3, H, W)
        spatial_off, path_off = torch.split(offset, [2, 1], dim=1)

        # 2. Bilinear sample dengan offset spasial
        # dtype=dtype WAJIB: tanpa ini, torch.tensor([...]) default ke fp32,
        # dan saat validasi Ultralytics menjalankan model via .half() (bukan
        # lewat autocast() yang otomatis menyamakan tipe), promosi Half->fp32
        # di sini membuat 'pos' jadi fp32 sementara 'x' tetap Half -> crash
        # grid_sample ("HalfTensor does not equal FloatTensor").
        off_range = torch.tensor([1.0 / max(H - 1, 1), 1.0 / max(W - 1, 1)],
                                 dtype=dtype, device=device).reshape(1, 2, 1, 1)
        spatial_off = spatial_off.tanh().mul(off_range)
        spatial_off = rearrange(spatial_off, 'b p h w -> b h w p').contiguous()
        ref = self._ref_grid(H, W, B, dtype, device)
        pos = ref + spatial_off  # (B, H, W, 2)

        # autocast(enabled=False) + cast eksplisit ke fp32: pertahanan lapis kedua.
        # grid_sample sensitif terhadap kecocokan dtype input vs grid; memaksa
        # fp32 di sini membuatnya kebal terhadap perbedaan precision-state
        # antara training (autocast) vs validasi (model.half()) Ultralytics.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_sampled = F.grid_sample(
                x.float(), pos[..., (1, 0)].float(),  # (y, x) -> (x, y) untuk grid_sample
                mode='bilinear', align_corners=True
            ).to(dtype)  # (B, C, H, W)
        x_flat = x_sampled.flatten(2)  # (B, C, N)

        # 3. Sort by path position
        path_off = path_off.tanh().flatten(1)  # (B, N)
        path_ref = self._path_ref(N, B, dtype, device)  # (B, N)
        path_pos = path_off + path_ref  # (B, N)

        indices = torch.argsort(path_pos, dim=-1)  # (B, N)
        idx_expand = indices.unsqueeze(1).expand(-1, C, -1)
        x_hard = torch.gather(x_flat, 2, idx_expand)  # (B, C, N) permutasi eksak

        # Straight-through estimator: argsort tidak diferensiabel, sehingga tanpa
        # ini path offset (Δp) tidak pernah menerima gradien dan scan path tidak
        # belajar. Nilai forward tidak berubah karena (x_soft - x_soft.detach()) = 0.
        sorted_pos = torch.gather(path_pos, 1, indices)  # (B, N) diferensiabel
        grid = torch.stack([sorted_pos, torch.zeros_like(sorted_pos)], -1).unsqueeze(1)
        with torch.autocast(device_type=x_flat.device.type, enabled=False):
            x_soft = F.grid_sample(x_flat.float().unsqueeze(2), grid.float(), mode='bilinear',
                                   padding_mode='border', align_corners=True).squeeze(2).to(x_flat.dtype)
        x_deformed = x_hard + (x_soft - x_soft.detach())  # (B, C, N)

        return x_deformed, indices

    @staticmethod
    def reverse(x_seq: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Un-sort: kembalikan urutan token ke posisi spatial asli."""
        B, C, N = x_seq.shape
        index_re = torch.zeros_like(indices)
        index_re.scatter_(1, indices,
                          torch.arange(N, device=x_seq.device).unsqueeze(0).expand(B, -1))
        return torch.gather(x_seq, 2, index_re.unsqueeze(1).expand(-1, C, -1))


# ============================================================================
#  CrossScan_Mixed — pengganti CrossScan dengan mode konfigurabel
# ============================================================================

def cross_scan_axial(x: torch.Tensor) -> torch.Tensor:
    """Scan 4 arah biasa (CrossScan SS2D asli) — versi non-autograd Function."""
    B, C, H, W = x.shape
    L = H * W
    xs = x.new_empty(B, 4, C, L)
    xs[:, 0] = x.flatten(2)                                    # row-major
    xs[:, 1] = x.transpose(2, 3).flatten(2)                    # col-major
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])             # reverse keduanya
    return xs


def cross_merge_axial(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Inverse dari cross_scan_axial."""
    B, K, C, L = ys.shape
    ys_re = ys.clone()
    ys_re[:, 2:4] = torch.flip(ys[:, 2:4], dims=[-1])
    ys_re[:, 1] = ys_re[:, 1].view(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    ys_re[:, 3] = ys_re[:, 3].view(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    return ys_re.sum(dim=1)


# ============================================================================
#  DeformSS2D — kelas utama
# ============================================================================

class DeformSS2D(SS2D):
    """
    Subclass SS2D dengan deformable scanning.

    deform_mode:
      "k3_replace"     : K=3 (forward + backward + deformable)        → ikut DefMamba
      "k5_augment"     : K=5 (4 SS2D + 1 deformable)                  → paling kaya
      "k4_replace_all" : K=4 (semua arah jadi deformable independen)  → paling agresif

    Stage menentukan kernel size offset network:
      stage 0 (H/4): K=9    stage 2 (H/16): K=5
      stage 1 (H/8): K=7    stage 3 (H/32): K=3
    """
    VALID_MODES = {"k3_replace", "k5_augment", "k4_replace_all"}

    def __init__(self, *args,
                 deform_mode: str = "k3_replace",
                 deform_stage: int = 0,
                 deform_kernel_size: int = None,
                 **kwargs):
        assert deform_mode in self.VALID_MODES, \
            f"deform_mode harus salah satu dari {self.VALID_MODES}, dapat: {deform_mode}"

        K_map = {"k3_replace": 3, "k5_augment": 5, "k4_replace_all": 4}
        self._target_K = K_map[deform_mode]

        super().__init__(*args, **kwargs)

        self.deform_mode = deform_mode
        self.deform_stage = deform_stage

        # Reinisialisasi parameter SSM dengan K baru
        if self._target_K != 4:
            self._reinit_ssm_params(self._target_K)

        # Offset network(s)
        ks_default = {0: 9, 1: 7, 2: 5, 3: 3}.get(deform_stage, 5)
        ks = deform_kernel_size if deform_kernel_size is not None else ks_default

        d_inner = self.x_proj_weight.shape[-1]

        n_deform = {"k3_replace": 1, "k5_augment": 1, "k4_replace_all": 4}[deform_mode]
        self.deform_scans = nn.ModuleList([
            DeformableScan(d_inner, kernel_size=ks) for _ in range(n_deform)
        ])

    def _reinit_ssm_params(self, K_new: int):
        """Reinit x_proj_weight, dt_projs, A_logs, Ds untuk K_new."""
        d_inner = self.x_proj_weight.shape[-1]
        d_state = self.A_logs.shape[-1]
        dt_rank = self.dt_projs_weight.shape[-1]

        self.K = K_new
        device = self.x_proj_weight.device

        new_x_proj = []
        for _ in range(K_new):
            lin = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False).to(device)
            new_x_proj.append(lin.weight)
        self.x_proj_weight = nn.Parameter(torch.stack(new_x_proj, dim=0))

        self.dt_projs_weight = nn.Parameter(
            torch.randn((K_new, d_inner, dt_rank), device=device)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.randn((K_new, d_inner), device=device)
        )

        self.A_logs = nn.Parameter(
            torch.zeros((K_new * d_inner, d_state), device=device)
        )
        self.Ds = nn.Parameter(torch.ones((K_new * d_inner), device=device))

    def _build_xs(self, x: torch.Tensor):
        """Bangun input sequences (B, K, C, L) sesuai deform_mode."""
        B, C, H, W = x.shape
        deform_indices = []

        if self.deform_mode == "k3_replace":
            x_fwd = x.flatten(2)
            x_bwd = torch.flip(x_fwd, dims=[-1])
            x_def, idx = self.deform_scans[0](x)
            xs = torch.stack([x_fwd, x_bwd, x_def], dim=1)
            deform_indices.append((2, idx))

        elif self.deform_mode == "k5_augment":
            xs_ss2d = cross_scan_axial(x)
            x_def, idx = self.deform_scans[0](x)
            xs = torch.cat([xs_ss2d, x_def.unsqueeze(1)], dim=1)
            deform_indices.append((4, idx))

        elif self.deform_mode == "k4_replace_all":
            xs_list = []
            for i in range(4):
                x_def, idx = self.deform_scans[i](x)
                xs_list.append(x_def)
                deform_indices.append((i, idx))
            xs = torch.stack(xs_list, dim=1)

        return xs, deform_indices

    def _merge_ys(self, ys: torch.Tensor, deform_indices, H: int, W: int):
        """Inverse dari _build_xs. (B, K, C, L) → (B, C, L)"""
        B, K, C, L = ys.shape

        if self.deform_mode == "k3_replace":
            y_fwd = ys[:, 0]
            y_bwd = torch.flip(ys[:, 1], dims=[-1])
            y_def = DeformableScan.reverse(ys[:, 2], deform_indices[0][1])
            return y_fwd + y_bwd + y_def

        elif self.deform_mode == "k5_augment":
            y_ss2d = cross_merge_axial(ys[:, :4], H, W)
            y_def = DeformableScan.reverse(ys[:, 4], deform_indices[0][1])
            return y_ss2d + y_def

        elif self.deform_mode == "k4_replace_all":
            y_total = torch.zeros(B, C, L, device=ys.device, dtype=ys.dtype)
            for slot_idx, idx in deform_indices:
                y_total = y_total + DeformableScan.reverse(ys[:, slot_idx], idx)
            return y_total

    def forward_corev2(self, x: torch.Tensor, channel_first=False,
                        SelectiveScan=SelectiveScanCore,
                        cross_selective_scan=None,
                        force_fp32=None):
        """Override forward_corev2 untuk pakai deformable scanning."""
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.ssm_low_rank:
            x = self.in_rank(x)

        B, D, H, W = x.shape
        L = H * W
        K = self._target_K

        # LANGKAH 1: Build xs sesuai deform_mode
        xs, deform_indices = self._build_xs(x)

        # LANGKAH 2: Project untuk SSM
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        xs_flat = xs.view(B, K * D, L)
        dts_flat = dts.contiguous().view(B, K * D, L)
        As = -torch.exp(self.A_logs.float())
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        Ds = self.Ds.float()
        delta_bias = self.dt_projs_bias.view(-1).float()

        if force_fp32:
            xs_flat = xs_flat.float()
            dts_flat = dts_flat.float()
            Bs = Bs.float()
            Cs = Cs.float()

        # LANGKAH 3: SSM
        ys = SelectiveScanCore.apply(
            xs_flat, dts_flat, As, Bs, Cs, Ds,
            delta_bias, True,  # delta_softplus=True
        ).view(B, K, D, L)

        # LANGKAH 4: Merge
        y = self._merge_ys(ys, deform_indices, H, W)
        y = y.view(B, D, H, W)

        # Out norm + permute ke (B, H, W, D)
        y = y.permute(0, 2, 3, 1).contiguous()
        if self.out_norm is not None:
            y = self.out_norm(y)

        if self.ssm_low_rank:
            y = self.out_rank(y)

        return y


# ============================================================================
#  Test cepat (jalankan untuk verifikasi)
# ============================================================================

# ============================================================================
#  Block-level subclasses — inilah yang didaftarkan di tasks.py & dipanggil YAML
# ============================================================================

class DeformVSSBlock(VSSBlock):
    """VSSBlock (backbone) dengan SS2D -> DeformSS2D.

    ssm_ratio diturunkan dari default 2.0 ke 1.4 untuk pencocokan kapasitas
    terhadap B2 (target +/-5%). VSSBlock.__init__ meneruskan ssm_ratio secara
    eksplisit ke self.SSM_CLASS(...), sehingga override harus terjadi di
    level __init__ ini, bukan lewat default di dalam partial SSM_CLASS
    (default partial akan tertimpa oleh nilai eksplisit dari VSSBlock).
    Hasil verifikasi (scale T, DSF-YOLO nc=8): B2=5.987M, P1=5.931M (-0.9%).
    """
    SSM_CLASS = partial(DeformSS2D, deform_mode="k3_replace", deform_kernel_size=5)

    def __init__(self, *args, ssm_ratio=2.0, **kwargs):
        super().__init__(*args, ssm_ratio=1.4, **kwargs)


class DeformXSSBlock(XSSBlock):
    """XSSBlock (neck) dengan SS2D -> DeformSS2D.

    ssm_ratio diturunkan dari default 2.0 ke 1.4 untuk pencocokan kapasitas
    terhadap B2 (target +/-5%). Lihat catatan di DeformVSSBlock di atas.
    """
    SSM_CLASS = partial(DeformSS2D, deform_mode="k3_replace", deform_kernel_size=5)

    def __init__(self, *args, ssm_ratio=2.0, **kwargs):
        super().__init__(*args, ssm_ratio=1.4, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, D, H, W = 2, 128, 16, 16

    for mode in ["k3_replace", "k5_augment", "k4_replace_all"]:
        print(f"\n=== Testing {mode} ===")
        x = torch.randn(B, D, H, W, device=device, requires_grad=True)  # NCHW
        m = DeformSS2D(d_model=D, deform_mode=mode, deform_stage=2).to(device)
        n_params = sum(p.numel() for p in m.parameters())
        y = m(x)
        loss = y.float().sum()
        loss.backward()
        print(f"  Input  : {tuple(x.shape)}")
        print(f"  Output : {tuple(y.shape)}")
        print(f"  Params : {n_params/1e6:.3f}M")
        print(f"  Grad   : {'OK' if x.grad is not None else 'FAIL'}")
        x.grad = None
