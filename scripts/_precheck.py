"""训练前预检脚本。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

ok = True

def check(label, passed, detail=""):
    global ok
    status = "OK  " if passed else "FAIL"
    if not passed:
        ok = False
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")

print()
print("=" * 60)
print("  PCaSAM-3D-ProFound  Pre-Training Check")
print("=" * 60)

# ── GPU ──────────────────────────────────────────────────────
print("\n[GPU]")
cuda_ok = torch.cuda.is_available()
check("CUDA available", cuda_ok)
if cuda_ok:
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check("GPU detected", True, f"{name}  {vram:.1f} GB VRAM")
    check("VRAM >= 16 GB", vram >= 16, f"{vram:.1f} GB")

# ── 依赖 ─────────────────────────────────────────────────────
print("\n[Dependencies]")
for pkg in ["numpy", "torch", "yaml", "tqdm", "scipy"]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "?")
        check(pkg, True, ver)
    except ImportError:
        check(pkg, False, "NOT INSTALLED")

# ── 权重文件 ─────────────────────────────────────────────────
print("\n[Checkpoints]")
profound_ckpt = Path("../ProFound/checkpoint/checkpoint-799 1.pth")
sam_ckpt = Path("../SAM-Med3D/ckpt/sam_med3d_turbo.pth")
check("ProFound checkpoint", profound_ckpt.is_file(), str(profound_ckpt))
check("SAM-Med3D checkpoint", sam_ckpt.is_file(), str(sam_ckpt))

# ── 数据 ─────────────────────────────────────────────────────
print("\n[Data]")
data_root = Path("../picai_preprocessing_project/data/processed/picai_profound_prompt_v2/all")
check("Data root exists", data_root.is_dir(), str(data_root))
if data_root.is_dir():
    npz_count = len(list(data_root.glob("*.npz")))
    check("NPZ count >= 1500", npz_count >= 1500, f"{npz_count} files")

split_root = Path("data/splits/5fold")
check("5-fold splits exist", split_root.is_dir())
if split_root.is_dir():
    for fold in range(5):
        train_f = split_root / f"fold_{fold}/train.txt"
        val_f   = split_root / f"fold_{fold}/val.txt"
        if train_f.is_file() and val_f.is_file():
            n_train = len(train_f.read_text().splitlines())
            n_val   = len(val_f.read_text().splitlines())
            check(f"fold_{fold}", True, f"train={n_train}  val={n_val}")
        else:
            check(f"fold_{fold}", False, "split files missing")

# ── 配置 ─────────────────────────────────────────────────────
print("\n[Config]")
try:
    from src.utils.config_utils import load_config
    cfg = load_config("configs/train_pcasam3d_profound.yaml")
    exp = cfg["logging"]["experiment_name"]
    out = cfg["logging"]["output_root"]
    check("Config loads", True, "configs/train_pcasam3d_profound.yaml")
    check("experiment_name", exp.startswith("pcasam3d_"), exp)
    check("output_root", out == "outputs", out)
    check("batch_size >= 1", cfg["data"]["batch_size"] >= 1,
          str(cfg["data"]["batch_size"]))
    check("amp enabled", cfg["training"]["amp"] is True)
except Exception as e:
    check("Config loads", False, str(e))

# ── 模델 forward smoke test ───────────────────────────────────
print("\n[Model smoke test]")
try:
    from src.models.pcasam3d_profound import build_pcasam3d_profound
    from src.models.pcasam3d_profound.pcasam3d_loss import build_pcasam3d_loss
    from src.utils.config_utils import load_config
    cfg = load_config("configs/train_pcasam3d_profound.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pcasam3d_profound(cfg).to(device)
    loss_fn = build_pcasam3d_loss(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    check("Model builds", True, f"{n_train:.1f}M trainable params")

    model.eval()
    x = torch.randn(1, 3, 128, 128, 128, device=device)
    label = torch.zeros(1, 1, 128, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
        losses = loss_fn(out, label)
    check("Forward pass", True,
          f"refined={out['refined_logits'].shape}  loss={losses['total_loss'].item():.3f}")
except Exception as e:
    check("Model smoke test", False, str(e))

# ── 结果 ─────────────────────────────────────────────────────
print()
print("=" * 60)
if ok:
    print("  RESULT: ALL CHECKS PASSED — ready to train!")
else:
    print("  RESULT: SOME CHECKS FAILED — fix before training.")
print("=" * 60)
print()
sys.exit(0 if ok else 1)
