# File-by-file Changes from Upstream DuST

This document lists every difference from the [upstream DuST repository](https://github.com/datadrivenwheels/DUST). The [main README](../README.md) explains the design. This file is the reference for anyone comparing the two repositories line by line.

| File in this repo | Upstream file | Status |
| --- | --- | --- |
| `models/dinov3_video_encoder.py` | `models/video_swin_transformer.py` | **New**, replaces the 3D Swin encoder |
| `models/SwinTransformer_cls.py`, `models/SwinTransformer_fusion.py` | same names | Copied unchanged |
| `models/__init__.py` | — | New: exports the encoder |
| `step_dinov3_3d.py` | `step_fusion_3d.py` | New model, new loss, same loop |
| `step_dinov3_3d_get_feature.py` | `step_fusion_3d_get_feature.py` | New model, feature-order bug fixed |
| `step_fusion_1d.py` | `step_fusion_1d.py` | Logic unchanged; paths and checkpointing |
| `step_fusion.py` | `step_fusion.py` | Logic unchanged; `embed_dim` bug fixed; checkpointing |
| `prepare_uah_data.py` | `gen_data.py` (toy data) | **New**: cleans and converts UAH-DriveSet |
| `notebooks/run_colab.ipynb` | — | New: runs Stages 0–4 on Colab |
| `visualization/` | — | New: data and training figures |
| `docs/README_DATA.md` | — | New: dataset documentation (Vietnamese) |
| — | `models/video_swin_transformer.py`, `gen_data.py` | No longer used |

## 1. `models/dinov3_video_encoder.py`

`DINOv3VideoEncoder` keeps the interface of `SwinTransformer3D`:

| Method | Input | Output |
| --- | --- | --- |
| `forward(x)` | `(B, 3, D, H, W)` | logits `(B, num_classes)` |
| `get_vid_feature(x)` | `(B, 3, D, H, W)` | feature `(B, 1024)` |

```text
video (B, 3, 77, 224, 224)
  -> permute, merge into B*77 frames
  -> DINOv3 ViT-S/16 in chunks of 16 frames, CLS token        (B, 77, 384)
  -> + temporal position embedding, prepend temporal CLS      (B, 78, 384)
  -> nn.TransformerEncoder (2 layers, 6 heads, FFN 1536, GELU)
  -> temporal CLS -> LayerNorm -> Linear(384, 1024)           (B, 1024)  = get_vid_feature
  -> Linear(1024, num_classes)                                (B, 4)     = forward
```

- `global_pool='token'` selects the CLS token. timm averages patch tokens for DINOv3 by default.
- `freeze_backbone=True` sets `requires_grad=False`, runs the backbone under `torch.no_grad()` and keeps it in `eval()` mode. `freeze_backbone=False` (the Stage 1 default since the SupCon change) lets gradients flow into DINOv3.
- `weights_path` loads a local `model.safetensors` through timm's `pretrained_cfg_overlay`. If the path is wrong, the encoder raises an error instead of using random weights.
- If timm fails, the encoder falls back to `transformers.AutoModel`.
- `--dinov3_model` accepts larger DINOv3 variants. The number of temporal heads defaults to `hidden_size // 64`.

## 2. `step_dinov3_3d.py` (Stage 1)

**Kept from upstream:**

- `CombinedLRScheduler` and `evaluate_test_loss`
- AdamW (`lr=1e-3`, `betas=(0.9, 0.999)`, `weight_decay=0.02`)
- 100 epochs, `print_every=3`, the seed, pickle loading
- `final_3d.pth`, `val_loss.npy`, `val_acc.npy`

```diff
- from models.video_swin_transformer import SwinTransformer3D
+ from models.dinov3_video_encoder import DINOv3VideoEncoder

- model = SwinTransformer3D(num_classes=4, embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
-                           patch_size=(2,4,4), window_size=(8,7,7), pretrained=..., pretrained2d=True)
+ model = DINOv3VideoEncoder(num_classes=4, num_frames=args.num_frames, model_name=args.dinov3_model,
+                            output_dim=1024, temporal_layers=args.temporal_layers,
+                            temporal_heads=args.temporal_heads, chunk_size=args.chunk_size,
+                            freeze_backbone=args.freeze_backbone, weights_path=args.dinov3_weights or None)

- criterion = nn.CrossEntropyLoss()
- optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)
+ criterion_train = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
+ criterion_eval = nn.CrossEntropyLoss()                 # validation stays plain CE
+ param_groups = [{'params': head_params, 'lr': lr},
+                 {'params': backbone_params, 'lr': lr * args.backbone_lr_mult}]
+ optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9,0.999), weight_decay=0.02)

- outputs = model(inputs_3d)
- loss = criterion(outputs, labels)
+ with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
+     feat = model.get_vid_feature(inputs_3d)
+     logits = model.head(feat)
+     ce = criterion_train(logits.float(), labels)
+     sc = supcon_loss(feat, labels, args.supcon_temp)
+     loss = ce + args.supcon_weight * sc
```

The script adds three functions:

- `supcon_loss()` (Khosla et al., 2020), computed in float32;
- `backbone_grad_norm()`, logged every 50 batches;
- `backbone_drift()`, giving ‖W − W₀‖ / ‖W₀‖, logged at each evaluation and saved to `dino_drift.json`.

The default batch size is 16 (upstream: 9).

## 3. `step_dinov3_3d_get_feature.py` (Stage 3)

```diff
- from models.video_swin_transformer import SwinTransformer3D
+ from models.dinov3_video_encoder import DINOv3VideoEncoder

- video_train = torch.from_numpy(pd.read_pickle(args.path_video_train).astype('float32'))
+ video_train = torch.from_numpy(pd.read_pickle(args.path_video_train))   # cast per batch

- train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
+ train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False)   # keep order aligned with labels

- model.load_state_dict(torch.load(args.path_3d_model))
+ model.load_state_dict(torch.load(args.path_3d_model, map_location=device))  # clear error if the file is missing
```

- The encoder is built with `pretrained=False`, because the Stage 1 checkpoint already contains the DINOv3 weights.
- The default batch size is 8 (upstream: 80) to fit 77 frames through DINOv3.
- The output names and shapes are unchanged.

## 4. `step_fusion_1d.py` (Stage 2)

The training logic is identical to upstream:

- `embed_dim=256`, `depths=[2,2,6,2]`, `num_heads=[16]*4`, `window_size=8`
- Adam with `lr=3e-4`, dropped to `3e-5` at epoch 100
- 200 epochs, batch 2000

Only the import order, the default data path (`./data/`), automatic creation of output folders and checkpointing (Section 6) differ.

## 5. `step_fusion.py` (Stage 4)

```diff
  model = SwinTransformer_1D_fusion(seq_len=seq_len, in_chans=num_chl, num_classes=4, window_size=8,
                                    patch_size=1, num_heads=[16, 16, 16, 16], depths=[2, 2, 6, 2],
-                                   embed_dim=128)
+                                   embed_dim=256)   # match Stage 2
```

Paths are joined with `os.path.join`, and output folders are created automatically. `evaluate_test_loss` now receives `device` as an argument instead of reading a global.

## 6. Checkpointing (Stages 1, 2, 4)

Upstream writes one `.pth` per evaluation. For the 1D and fusion models (~760 MB each) that comes to ~150 GB per stage.

```diff
- torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')
+ if save_ckpt:   # only with --save_all_epochs
+     torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')
+ ...
+ if not args.save_all_epochs and test_loss_item < best_val_loss:
+     torch.save(model.state_dict(), outpath+'best_1d.pth')   # overwritten, one file
```

By default only `best_*.pth` (lowest validation loss) and `final_*.pth` (last epoch) are kept. `--save_all_epochs` restores the upstream behaviour.

## 7. Upstream bugs

| Bug | Effect | Fix |
| --- | --- | --- |
| Stage 4 uses `embed_dim=128` while Stage 2 uses 256 (upstream commit `eec315b`) | Stage 4 cannot load the Stage 2 weights | `embed_dim=256` |
| Stage 3 extracts train features with `shuffle=True` | Video features are misaligned with kinematics and labels in Stage 4 | `shuffle=False` |
| A checkpoint is saved at every evaluation | ~150 GB per stage | best + final only |
| `logs/` and `saved_models/` are missing in a fresh folder | `logging` fails at start-up | Folders are created automatically |
