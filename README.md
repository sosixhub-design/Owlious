# Nook motion model (MotionFlow v5.1)

## 1. Train / load weights (on your PC or Colab)
    import torch, motion_model as mm
    m = mm.MotionFlow()
    m = mm.load_weights(m, "your_checkpoint.pt")
    mm.export_onnx(m, "motion", height=288, width=512)   # sizes must be multiples of 32
This writes `motion_flow.onnx` (used by Nook) and `motion_interp.onnx` (optional).
NOTE: motion_model.py has NO trained weights. Untrained, it outputs noise, so Nook's
tracking/slow-mo will look wrong until you train it.

## 2. Host on GitHub Pages
Put `models.json`, `motion_flow.onnx` (and `motion_interp.onnx`) in a repo, turn on
Pages. Link example: `https://YOURNAME.github.io/nook-models/`

## 3. In Nook
Select a video clip or a text/shape -> **Motion** tool -> paste the link -> Connect.
- text/shape over a video -> **Follow the motion** (position keyframes)
- video clip -> **Slow motion x2/x4/x8** (AI in-between frames, new clip)

ONNX inputs `I0`,`I1` = float32 [1,3,288,512] in 0..1. Outputs: `flow_f`, `flow_b`,
`occ_mask`, `motion_mask`.
