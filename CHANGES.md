# motion_model.py v5.0 -> v5.1 fixes
1. WindowAttention: output reshape order was wrong (scrambled the attention result). Fixed + verified round-trip.
2. Frame sizes not divisible by 32 crashed (flow up/down mismatch, UNet skips). Now padded and cropped.
3. Duplicate `__init__` removed.
4. CorrBlock used torch.roll (matches wrapped from the opposite edge). Now zero-padded shifts.
5. forward_tiled: scene-cut/static decision was per tile (seams). Now once per frame; ramp never 0.
6. forward_flow: flows returned at original size/units; optional max_side for speed.
7. ObjectTracker: downscaled flow (fast), box size kept, clamped.
8. New: load_weights(), export_onnx() (pure `_core` graph, no data-dependent python branches).
Not tested: no PyTorch in my sandbox. Please run a forward pass + export_onnx once.
