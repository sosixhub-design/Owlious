"""
motion_model.py — MotionFlow v5: motion-aware interpolation + flow + tracking, hardened.

v5 — ROBUSTNESS (anti-lag / anti-crash / anti frame-errors):
  - CACHED SAMPLING GRIDS: warp/adaptive_warp/trajectory smoother reuse grid
    buffers per (H,W,device,dtype) instead of rebuilding them every call.
  - INPUT SANITIZATION: uint8/any-float coercion, [0,1] clamping, NaN/Inf
    cleaning, t clamped to [eps, 1-eps], clear errors on bad shapes.
  - STATIC-FRAME SHORTCUT: identical inputs skip the network entirely.
  - SCENE-CUT GUARD: drastically different inputs fall back to a temporal
    cross-dissolve instead of warped garbage (safe_mode, on by default).
  - MIN-SIZE GUARD: tiny inputs (<64px) are upsampled, processed, restored.
  - OUTPUT NaN GUARD: synthesized frames can never emit NaN/Inf pixels.
  - TILED INFERENCE: forward_tiled() for 4K/8K frames — prevents OOM.

v4 made the model explicitly MOTION-AWARE ("detect moving pixels, make
motion smooth"):
  1. MOTION DETECTION HEAD — per-pixel moving-pixel mask M, predicted from
     the bidirectional flows + inputs. Moving pixels are fully synthesized by
     the flow pipeline; STATIC pixels are passed through as a plain temporal
     cross-dissolve. Backgrounds, text, and still objects never warp —
     kills the "wobble/ghosting on static content" artifact class entirely.
  2. TRAJECTORY SMOOTHER — moving pixels are filtered ALONG their motion
     trajectory (5 taps placed on the flow vector, learned per-pixel
     weights). Motion looks fluid and denoised without smearing across edges.

Retained from v3: correlation cost volume, deformable kernel synthesis,
windowed self-attention, bidirectional flows, occlusion mask, temporal
offset + quadratic curvature, 4 coarse-to-fine stages, dual-frame context.

v5.1 — BUG FIXES (found in review, see CHANGES.md):
  - WindowAttention output was reshaped in the wrong order -> scrambled features
  - sizes not a multiple of 32 crashed (odd / non-32 frames): now padded + cropped
  - duplicate __init__ removed
  - CorrBlock used torch.roll (wrapped pixels from the opposite edge): now zero-shift
  - tiled inference decided scene-cut per tile (visible seams): now per frame
  - forward_flow flows are returned in the ORIGINAL pixel units/size
  - ObjectTracker runs flow on a downscaled copy (fast) and clamps the box
  - export_onnx() for the browser (Nook) + load_weights() helper

Same API: forward(I0, I1, t) -> interpolated frame at time t.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "5.1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_GRID_CACHE: dict = {}


def _base_grid(B, H, W, device, dtype):
    """Cached normalized [-1,1] sampling grid — rebuilt only per new shape."""
    key = (H, W, str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W)
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W)
        g = torch.stack((xx, yy), dim=-1)
        if len(_GRID_CACHE) > 64:      # bound memory
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = g
    return g.expand(B, -1, -1, -1)


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp x by flow (B,2,H,W) using grid_sample."""
    B, _, H, W = x.shape
    grid = _base_grid(B, H, W, x.device, x.dtype)
    grid = grid + flow.permute(0, 2, 3, 1) * torch.tensor(
        [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=x.device, dtype=x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def adaptive_warp(x, flow, off, mod):
    """
    Modulated deformable re-sampling of x.
      flow : (B,2,H,W) base backward flow
      off  : (B,18,H,W) per-tap offsets for the 3x3 kernel
      mod  : (B,9,H,W) per-tap modulation (softmaxed)
    """
    B, _, H, W = x.shape
    grid0 = _base_grid(B, H, W, x.device, x.dtype)
    norm = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=x.device, dtype=x.dtype)
    flowg = flow.permute(0, 2, 3, 1) * norm
    taps = mod.softmax(dim=1)          # (B,9,H,W)
    out = torch.zeros_like(x)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            delta = off[:, 2 * k:2 * k + 2].permute(0, 2, 3, 1) * norm
            g = grid0 + flowg + delta
            out = out + taps[:, k:k + 1] * \
                F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)
            k += 1
    return out


class Head(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.c(x))


class WindowAttention(nn.Module):
    """Non-overlapping 8x8 window multi-head self-attention (cheap at any res)."""

    def __init__(self, c, heads=4, win=8):
        super().__init__()
        self.h, self.win = heads, win
        self.norm = nn.GroupNorm(4, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.win
        x = self.norm(x)
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]
        qkv = self.qkv(x).reshape(B, 3, self.h, C // self.h, Hp // w, w, Wp // w, w)
        q, k, v = qkv.unbind(1)   # each: (B,h,C/h,nWh,w,nWw,w)
        perm = (0, 3, 5, 1, 4, 6, 2)
        q = q.permute(perm).flatten(4, 5)   # B,nWh,nWw,h,ww,C/h
        k = k.permute(perm).flatten(4, 5)
        v = v.permute(perm).flatten(4, 5)
        att = F.scaled_dot_product_attention(q, k, v)          # (B,nWh,nWw,h,ww,C/h)
        # att dims: (B, nWh, nWw, heads, wy*wx, C/h) -> split the window, then back to (B, C, Hp, Wp)
        att = att.reshape(B, Hp // w, Wp // w, self.h, w, w, C // self.h)
        att = att.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        if pad_h or pad_w:
            att = att[..., :H, :W]
        return x[..., :H, :W] + self.proj(att)


class CorrBlock(nn.Module):
    """Local correlation cost volume for large-motion matching (coarse scale)."""

    def __init__(self, r=4):
        super().__init__()
        self.r = r

    def forward(self, f0, f1):
        B, C, H, W = f0.shape
        r = self.r
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)
        vol = []
        f1p = F.pad(f1, (r, r, r, r))                  # zero border: no wrap-around matches
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                s1 = f1p[..., r + dy:r + dy + H, r + dx:r + dx + W]
                vol.append((f0 * s1).sum(1, keepdim=True))
        return torch.cat(vol, 1)  # (B, (2r+1)^2, H, W)


class FlowBlock(nn.Module):
    """
    One coarse-to-fine stage. Predicts 7 channels:
    forward flow (0->1), backward flow (1->0), occlusion mask,
    temporal offset, quadratic curvature map.
    Prev flows (4ch) concatenated inside the block when provided.
    """

    def __init__(self, in_ch, c=192):
        super().__init__()
        self.conv0 = nn.Sequential(Head(in_ch, c // 2), Head(c // 2, c))
        self.convblock = nn.Sequential(
            Head(c, c), Head(c, c), Head(c, c), Head(c, c), Head(c, c))
        self.lastconv = nn.ConvTranspose2d(c, 7, 4, 2, 1)

    def forward(self, x, flow_f=None, flow_b=None):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        if flow_f is not None:
            flow_f = F.interpolate(flow_f, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            flow_b = F.interpolate(flow_b, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            x = torch.cat((x, flow_f, flow_b), 1)
        feat = self.convblock(self.conv0(x))
        tmp = self.lastconv(feat)
        nf, nb, mask, t_off, curv = tmp[:, 0:2], tmp[:, 2:4], tmp[:, 4:5], tmp[:, 5:6], tmp[:, 6:7]
        if flow_f is not None:
            up = lambda f: F.interpolate(f, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
            nf, nb = nf + up(flow_f), nb + up(flow_b)
        return nf, nb, torch.sigmoid(mask), torch.tanh(t_off) * 0.5, torch.tanh(curv) * 0.5


class TrajectorySmoother(nn.Module):
    """
    Flow-guided smoothing along motion trajectories.
    Samples the frame at 5 points along its own flow vector and blends them
    with learned, per-pixel weights -> smooths moving content without
    blurring across edges (filter is parallel to motion, not across it).
    """

    def __init__(self, taps=5):
        super().__init__()
        self.taps = taps
        self.weight_net = Head(3 + 2 + 1, taps)   # frame + flow + motion mask -> tap weights

    def forward(self, x, flow, m_mask):
        B, _, H, W = x.shape
        grid0 = _base_grid(B, H, W, x.device, x.dtype)
        norm = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=x.device, dtype=x.dtype)
        steps = torch.linspace(-0.5, 0.5, self.taps, device=x.device, dtype=x.dtype)
        w = self.weight_net(torch.cat((x, flow, m_mask), 1)).softmax(dim=1)  # (B,taps,H,W)
        out = torch.zeros_like(x)
        for k in range(self.taps):
            g = grid0 + flow.permute(0, 2, 3, 1) * norm * steps[k]
            out = out + w[:, k:k + 1] * F.grid_sample(
                x, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out


class MotionHead(nn.Module):
    """Per-pixel moving-pixel detector: 1 = moving (synthesize), 0 = static (pass-through)."""

    def __init__(self, c=48):
        super().__init__()
        self.net = nn.Sequential(
            Head(3 + 3 + 1 + 2 + 2, c), Head(c, c), Head(c, 1, act=False))

    def forward(self, I0, I1, t_eff, ff, fb):
        x = torch.cat((I0, I1, t_eff, ff, fb), 1)
        return torch.sigmoid(self.net(x))   # (B,1,H,W)


class ContextNet(nn.Module):
    """Multi-scale context features from one frame, warped by its own flow."""

    def __init__(self, c=64):
        super().__init__()
        self.c1 = Head(3, c // 2)
        self.c2 = Head(c // 2 + 3, c)
        self.c3 = Head(c + 3, c * 2)

    def forward(self, img, flow):
        f1 = self.c1(img)
        img2 = warp(img, flow)
        f2_in = torch.cat((F.interpolate(f1, scale_factor=0.5, mode="bilinear", align_corners=False),
                           F.interpolate(img2, scale_factor=0.5, mode="bilinear", align_corners=False)), 1)
        f2 = self.c2(f2_in)
        f3_in = torch.cat((F.interpolate(f2, scale_factor=0.5, mode="bilinear", align_corners=False),
                           F.interpolate(img2, scale_factor=0.25, mode="bilinear", align_corners=False)), 1)
        f3 = self.c3(f3_in)
        return f1, f2, f3


class RefineUNet(nn.Module):
    """
    Dual-frame context UNet. Adds v3:
      - windowed self-attention at the bottleneck (disocclusion coherence)
      - deformable kernel head (27ch: 18 offsets + 9 modulation, applied to both
        warped frames) replacing plain bilinear as the final sampler.
    """

    def __init__(self, c=64):
        super().__init__()
        self.enc0 = Head(3 + 3 + 1 + 2 * (c // 2), c // 2)
        self.enc1 = Head(c // 2, c)
        self.enc2 = Head(c, c * 2)
        self.proj2 = nn.Conv2d(2 * c, c, 1)
        self.proj3 = nn.Conv2d(4 * c, c * 2, 1)
        self.attn = WindowAttention(c * 2, heads=4, win=8)
        self.dec2 = Head(c * 2, c)
        self.up1 = nn.ConvTranspose2d(c, c // 2, 4, 2, 1)
        self.dec1 = Head(c + c // 2, c // 2)
        self.up0 = nn.ConvTranspose2d(c // 2, c // 4, 4, 2, 1)
        self.dec0 = Head(c // 2 + c // 4, c // 4)
        self.last = nn.Conv2d(c // 4, 3 + 36 + 9, 3, 1, 1)  # 3 residual + 2x18 offsets + 9 modulation

    def forward(self, merged, ctx0, ctx1):
        f1a, f2a, f3a = ctx0
        f1b, f2b, f3b = ctx1
        e0 = self.enc0(torch.cat((merged, f1a, f1b), 1))
        e1 = self.enc1(F.interpolate(e0, scale_factor=0.5, mode="bilinear", align_corners=False))
        e1 = e1 + self.proj2(torch.cat((f2a, f2b), 1))
        e2 = self.enc2(F.interpolate(e1, scale_factor=0.5, mode="bilinear", align_corners=False))
        e2 = e2 + self.proj3(torch.cat((f3a, f3b), 1))
        e2 = self.attn(e2)
        d2 = self.dec2(e2)
        u1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((u1, e1), 1))
        u0 = F.interpolate(self.up0(d1), size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat((u0, e0), 1))
        out = self.last(d0)
        return out[:, :3], out[:, 3:39], out[:, 39:48]  # residual, offsets(36), modulation(9)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class MotionFlow(nn.Module):
    """
    Arbitrary-time frame interpolation (v3).

    forward(I0, I1, t):
        I0, I1 : (B,3,H,W) float in [0,1]
        t      : (B,) float in (0,1)
    returns I_t (B,3,H,W)
    """

    def __init__(self, c=192, ctx_c=64):
        super().__init__()
        self.corr = CorrBlock(r=4)
        self.corr_proj = Head(81, 64)   # compress cost volume into feature channels
        self.block0 = FlowBlock(3 + 3 + 1 + 64, c)
        blk_in = 3 + 3 + 1 + 2 + 2 + 3 + 3 + 1 + 4
        self.block1 = FlowBlock(blk_in, c)
        self.block2 = FlowBlock(blk_in, c)
        self.block3 = FlowBlock(blk_in, c)
        self.context = ContextNet(ctx_c)
        self.refine = RefineUNet(ctx_c)
        self.motion_head = MotionHead()
        self.smoother = TrajectorySmoother(taps=5)
        # robustness knobs
        self.safe_mode = True        # scene-cut guard + static shortcut
        self.cut_threshold = 0.35    # mean |I0-I1| above this => treat as cut

    @staticmethod
    def _sanitize(I0, I1, t):
        """Coerce dtypes/shapes, clamp ranges, kill NaNs before they propagate."""
        if not torch.is_tensor(I0):
            I0 = torch.as_tensor(I0)
        if not torch.is_tensor(I1):
            I1 = torch.as_tensor(I1)
        if I0.dim() == 3:
            I0 = I0.unsqueeze(0)
        if I1.dim() == 3:
            I1 = I1.unsqueeze(0)
        if I0.shape != I1.shape:
            raise ValueError(f"I0/I1 shape mismatch: {tuple(I0.shape)} vs {tuple(I1.shape)}")
        if I0.shape[-2] < 4 or I0.shape[-1] < 4:
            raise ValueError("frames must be at least 4x4 pixels")
        if I0.shape[1] not in (1, 3):
            raise ValueError("frames must have 1 or 3 channels")
        if I0.dtype == torch.uint8:
            I0, I1 = I0.float() / 255.0, I1.float() / 255.0
        I0 = torch.nan_to_num(I0.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        I1 = torch.nan_to_num(I1.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        if I0.shape[1] == 1:
            I0, I1 = I0.repeat(1, 3, 1, 1), I1.repeat(1, 3, 1, 1)
        t = torch.as_tensor(t, device=I0.device, dtype=I0.dtype).reshape(-1).clamp(1e-3, 1 - 1e-3)
        if t.numel() == 1 and I0.shape[0] > 1:
            t = t.expand(I0.shape[0])
        if t.numel() != I0.shape[0]:
            raise ValueError(f"t must have 1 or B={I0.shape[0]} elements, got {t.numel()}")
        return I0, I1, t

    # ------------------------------------------------------------------ #
    # Pure tensor core: NO python branching on tensor values -> exportable
    # to ONNX. Input H and W must be multiples of 32 (see _pad32).
    # ------------------------------------------------------------------ #
    def _core(self, I0, I1, t):
        B = I0.shape[0]
        t_map = t.view(B, 1, 1, 1).expand(B, 1, *I0.shape[-2:])

        def eff_time(t_off, curv):
            return torch.clamp(t_map + t_off * t_map + curv * t_map * (1 - t_map), 0.0, 1.0)

        def stage_input(I0s, I1s, t_eff, ff, fb, mask):
            w0 = warp(I0s, ff * t_eff.repeat(1, 2, 1, 1))
            w1 = warp(I1s, fb * (1 - t_eff).repeat(1, 2, 1, 1))
            return torch.cat((I0s, I1s, t_eff, ff, fb, w0, w1, mask), 1)

        # coarsest stage: correlation volume helps anchor large motion
        c0 = F.interpolate(torch.cat((I0, I1), 1), scale_factor=0.25, mode="bilinear", align_corners=False)
        f0 = self.corr_proj(self.corr(c0[:, :3], c0[:, 3:6]))
        x0 = torch.cat((I0, I1, t_map, F.interpolate(f0, size=I0.shape[-2:], mode="bilinear", align_corners=False)), 1)
        ff, fb, mask, t_off, curv = self.block0(x0)
        t_eff = eff_time(t_off, curv)
        for blk in (self.block1, self.block2, self.block3):
            x = stage_input(I0, I1, t_eff, ff, fb, mask)
            ff, fb, mask, t_off, curv = blk(x, ff, fb)
            t_eff = eff_time(t_off, curv)

        w0 = warp(I0, ff * t_eff.repeat(1, 2, 1, 1))
        w1 = warp(I1, fb * (1 - t_eff).repeat(1, 2, 1, 1))
        ctx0 = self.context(I0, ff)
        ctx1 = self.context(I1, fb)
        res, d_off, d_mod = self.refine(torch.cat((w0, w1, mask), 1), ctx0, ctx1)

        m_mask = self.motion_head(I0, I1, t_eff, ff, fb)                 # moving pixels
        I0s = I0 * (1 - m_mask) + self.smoother(I0, ff, m_mask) * m_mask  # smooth along trajectories
        I1s = I1 * (1 - m_mask) + self.smoother(I1, fb, m_mask) * m_mask
        dw0 = adaptive_warp(I0s, ff * t_eff.repeat(1, 2, 1, 1), d_off[:, :18], d_mod)
        dw1 = adaptive_warp(I1s, fb * (1 - t_eff).repeat(1, 2, 1, 1), d_off[:, 18:], d_mod)
        interp = torch.clamp(dw0 * mask + dw1 * (1 - mask) + res, 0, 1)
        static_blend = (1 - t_eff) * I0 + t_eff * I1
        out = torch.clamp(interp * m_mask + static_blend * (1 - m_mask), 0, 1)
        return out, ff, fb, mask, m_mask, t_eff

    @staticmethod
    def _pad32(x, mult=32, min_side=64):
        """Replicate-pad bottom/right so H and W are multiples of `mult` (and >= min_side)."""
        H, W = x.shape[-2:]
        Hp = max(min_side, -(-H // mult) * mult)
        Wp = max(min_side, -(-W // mult) * mult)
        if Hp != H or Wp != W:
            x = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
        return x

    def _shortcut(self, I0, I1, t):
        """Safe mode: returns an output frame when the network should be skipped, else None."""
        if not self.safe_mode:
            return None
        B = I0.shape[0]
        diff = (I0 - I1).abs().mean(dim=(1, 2, 3))
        static, cut = diff < 1e-4, diff > self.cut_threshold
        if bool(static.all() or cut.all()):
            tm = t.view(B, 1, 1, 1)
            return torch.where(static.view(B, 1, 1, 1), I0, (1 - tm) * I0 + tm * I1)
        return None

    def _forward_all(self, I0, I1, t, use_shortcut=True):
        """Full pipeline; returns (frame, aux dict with flows/masks) at the ORIGINAL size."""
        I0, I1, t = self._sanitize(I0, I1, t)
        B = I0.shape[0]
        H0, W0 = I0.shape[-2:]
        if use_shortcut:
            sc = self._shortcut(I0, I1, t)
            if sc is not None:
                z = torch.zeros(B, 2, H0, W0, device=I0.device, dtype=I0.dtype)
                o = torch.full((B, 1, H0, W0), 0.5, device=I0.device, dtype=I0.dtype)
                return torch.clamp(sc, 0, 1), {"flow_f": z, "flow_b": z, "occ_mask": o, "motion_mask": o,
                                               "t_eff": t.view(B, 1, 1, 1)}
        P0, P1 = self._pad32(I0), self._pad32(I1)
        out, ff, fb, mask, m_mask, t_eff = self._core(P0, P1, t)
        crop = lambda x: x[..., :H0, :W0]
        out = torch.nan_to_num(crop(out), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        return out, {"flow_f": crop(ff), "flow_b": crop(fb), "occ_mask": crop(mask),
                     "motion_mask": crop(m_mask), "t_eff": crop(t_eff)}

    @torch.no_grad()
    def forward_tiled(self, I0, I1, t, tile=768, overlap=32):
        """
        CRASH-SAFE inference for very large frames (4K/8K): runs the pipeline
        on overlapping tiles and blends seams with a linear ramp. The cut /
        duplicate-frame decision is made ONCE on the whole frame so tiles
        never disagree (no seams).
        """
        I0, I1, t = self._sanitize(I0, I1, t)
        B, _, H, W = I0.shape
        tile = max(64, (tile // 32) * 32)
        overlap = max(8, min(overlap, tile // 4))
        if H <= tile and W <= tile:
            return self._forward_all(I0, I1, t)[0]
        sc = self._shortcut(I0, I1, t)
        if sc is not None:
            return torch.clamp(sc, 0, 1)
        out = torch.zeros_like(I0)
        weight = torch.zeros(B, 1, H, W, device=I0.device, dtype=I0.dtype)
        stride = tile - overlap
        ramp = lambda n: (torch.arange(n, device=I0.device, dtype=I0.dtype) + 1) / (n + 1)   # never exactly 0
        ys = list(range(0, max(H - overlap, 1), stride))
        xs = list(range(0, max(W - overlap, 1), stride))
        for y in ys:
            for x in xs:
                y2, x2 = min(y + tile, H), min(x + tile, W)
                y1, x1 = max(y2 - tile, 0), max(x2 - tile, 0)
                sub, _ = self._forward_all(I0[..., y1:y2, x1:x2], I1[..., y1:y2, x1:x2], t, use_shortcut=False)
                wy = torch.ones(y2 - y1, device=I0.device, dtype=I0.dtype)
                wx = torch.ones(x2 - x1, device=I0.device, dtype=I0.dtype)
                if y1 > 0:
                    wy[:overlap] = ramp(overlap)
                if y2 < H:
                    wy[-overlap:] = ramp(overlap).flip(0)
                if x1 > 0:
                    wx[:overlap] = ramp(overlap)
                if x2 < W:
                    wx[-overlap:] = ramp(overlap).flip(0)
                w = (wy[:, None] * wx[None, :]).view(1, 1, y2 - y1, x2 - x1)
                out[..., y1:y2, x1:x2] += sub * w
                weight[..., y1:y2, x1:x2] += w
        return torch.clamp(out / weight.clamp(min=1e-6), 0, 1)

    def forward(self, I0, I1, t):
        """Frame interpolation: returns I_t (B,3,H,W)."""
        out, _ = self._forward_all(I0, I1, t)
        return out

    @torch.no_grad()
    def forward_flow(self, I0, I1, t=None, max_side=0):
        """
        OPTICAL FLOW between I0 and I1, returned at the ORIGINAL frame size and
        in ORIGINAL pixel units.  max_side>0 runs the network on a downscaled
        copy (much faster) and scales the flow back up.

        Returns (flow_f, flow_b, occ_mask, motion_mask, global_motion).
        global_motion is the median flow (B,2) - the dominant camera motion.
        """
        I0, I1, tt = self._sanitize(I0, I1, 0.5 if t is None else t)
        H, W = I0.shape[-2:]
        k = 1.0
        if max_side and max(H, W) > max_side:
            k = max_side / float(max(H, W))
            sz = (max(64, int(round(H * k))), max(64, int(round(W * k))))
            I0 = F.interpolate(I0, size=sz, mode="bilinear", align_corners=False)
            I1 = F.interpolate(I1, size=sz, mode="bilinear", align_corners=False)
        _, aux = self._forward_all(I0, I1, tt)
        ff, fb, om, mm = aux["flow_f"], aux["flow_b"], aux["occ_mask"], aux["motion_mask"]
        if k != 1.0:
            up = lambda x, sc=1.0: F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) * sc
            ff, fb, om, mm = up(ff, 1.0 / k), up(fb, 1.0 / k), up(om), up(mm)
        gm = ff.flatten(2).median(dim=2).values
        return ff, fb, om, mm, gm


# --------------------------------------------------------------------------- #
# Object motion tracking
# --------------------------------------------------------------------------- #
class ObjectTracker:
    """
    Tracks a selected object/person across frames so effects, text, and other
    layers can follow it. Box-based: give the init box on the first frame;
    each update shifts the box by the median flow inside it (robust to partial
    occlusion and to other motion in the scene).

        trk = ObjectTracker(model, frame0, box=(x1, y1, x2, y2))   # pixels
        for frame in frames[1:]:
            box = trk.update(frame)
        traj = trk.trajectory
    """

    def __init__(self, model, init_frame, box, max_side=384):
        self.model = model
        self.max_side = max_side
        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cpu")
        self.prev = self._prep(init_frame)
        self.size = tuple(self.prev.shape[-2:])
        x1, y1, x2, y2 = [float(v) for v in box]
        self.box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        self.trajectory = [tuple(self.box)]

    def _prep(self, frame):
        if not torch.is_tensor(frame):
            frame = torch.as_tensor(frame)
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        return frame[:, :3].to(self.device).float()

    @torch.no_grad()
    def update(self, frame):
        cur = self._prep(frame)
        if tuple(cur.shape[-2:]) != self.size:
            raise ValueError("every frame must have the same size as the first one")
        flow_f, _, _, mm, _ = self.model.forward_flow(self.prev, cur, max_side=self.max_side)
        h, w = flow_f.shape[-2:]
        x1, y1, x2, y2 = self.box
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))
        if x2c - x1c < 2 or y2c - y1c < 2:
            dx = dy = 0.0
        else:
            region = flow_f[0, :, y1c:y2c, x1c:x2c].reshape(2, -1)
            dx, dy = region.median(dim=1).values.tolist()
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = min(max(x1 + dx, -bw * 0.5), w - bw * 0.5), min(max(y1 + dy, -bh * 0.5), h - bh * 0.5)
        self.box = [x1, y1, x1 + bw, y1 + bh]
        self.prev = cur
        self.trajectory.append(tuple(self.box))
        return tuple(self.box)


@torch.no_grad()
def track_object(model, frames, box, max_side=384):
    """One-shot: track `box` (x1,y1,x2,y2) through `frames`. Returns one box per frame."""
    trk = ObjectTracker(model, frames[0], box, max_side=max_side)
    for f in frames[1:]:
        trk.update(f)
    return trk.trajectory


# --------------------------------------------------------------------------- #
# Weights + browser export
# --------------------------------------------------------------------------- #
def load_weights(model, path, device="cpu"):
    """Load a checkpoint (plain state_dict or {'model'/'state_dict': ...}); reports mismatches."""
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict):
        ck = ck.get("model", ck.get("state_dict", ck))
    ck = {k.replace("module.", "", 1): v for k, v in ck.items()}
    missing, unexpected = model.load_state_dict(ck, strict=False)
    if missing or unexpected:
        print(f"[motion_model] missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval().to(device)


class _OnnxWrap(nn.Module):
    """ONNX graph: (I0, I1, t) -> frame, fixed size multiple of 32. Frames in [0,1], NCHW float32."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, I0, I1, t):
        return self.m._core(I0, I1, t.reshape(-1).clamp(1e-3, 1 - 1e-3))[0]


class _OnnxFlowWrap(nn.Module):
    """ONNX graph: (I0, I1) -> flow_f, flow_b (B,2,H,W) in pixels of the model size, occ_mask, motion_mask."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, I0, I1):
        t = torch.full((I0.shape[0],), 0.5, device=I0.device, dtype=I0.dtype)
        _, ff, fb, mask, mm, _ = self.m._core(I0, I1, t)
        return ff, fb, mask, mm


def export_onnx(model, out_prefix="motion", height=288, width=512, opset=17):
    """
    Writes <prefix>_interp.onnx and <prefix>_flow.onnx for the Nook editor.
    height/width must be multiples of 32; the browser resizes frames to this size.
    (grid_sample needs opset >= 16.)
    """
    assert height % 32 == 0 and width % 32 == 0, "height/width must be multiples of 32"
    model = model.eval().cpu()
    a = torch.rand(1, 3, height, width)
    b = torch.rand(1, 3, height, width)
    t = torch.tensor([0.5])
    torch.onnx.export(_OnnxWrap(model), (a, b, t), f"{out_prefix}_interp.onnx", opset_version=opset,
                      input_names=["I0", "I1", "t"], output_names=["frame"], do_constant_folding=True)
    torch.onnx.export(_OnnxFlowWrap(model), (a, b), f"{out_prefix}_flow.onnx", opset_version=opset,
                      input_names=["I0", "I1"], output_names=["flow_f", "flow_b", "occ_mask", "motion_mask"], do_constant_folding=True)
    print(f"wrote {out_prefix}_interp.onnx and {out_prefix}_flow.onnx at {width}x{height}")


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class MotionFlowLoss(nn.Module):
    """L1 + cosine + flow/mask regularization, optional LapLoss for sharpness."""

    def __init__(self, use_lap=False):
        super().__init__()
        self.use_lap = use_lap

    def census(self, a, b):
        return F.l1_loss(a, b) + (1 - F.cosine_similarity(a.flatten(1), b.flatten(1)).mean())

    def lap(self, a, b):
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=a.device, dtype=a.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        la = F.conv2d(F.pad(a, (1, 1, 1, 1), mode="replicate"), k, groups=3)
        lb = F.conv2d(F.pad(b, (1, 1, 1, 1), mode="replicate"), k, groups=3)
        return F.l1_loss(la, lb)

    def forward(self, pred, gt, flow_f, flow_b, mask):
        l_img = self.census(pred, gt)
        if self.use_lap:
            l_img = l_img + 0.5 * self.lap(pred, gt)
        l_flow = (flow_f.abs().mean() + flow_b.abs().mean()) * 0.01
        l_mask = mask.abs().mean() * 0.01
        return l_img + l_flow + l_mask
