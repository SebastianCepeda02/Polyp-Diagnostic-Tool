import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# ===== Config por defecto =====
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET = (352, 352)  # tamaño esperado por el modelo
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _preprocess_bgr(img_bgr, size=TARGET, mean=MEAN, std=STD):
    """BGR -> RGB/float -> resize -> normalize -> tensor BCHW"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[:2]
    res = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    res = (res - mean) / std
    ten = torch.from_numpy(res.transpose(2, 0, 1)).unsqueeze(0).float()
    return ten, (h0, w0)

def _postprocess_prob(prob, orig_hw, bin_thresh=0.5):
    """(H,W) prob -> resize back -> [0..1] -> binaria 0/255"""
    h0, w0 = orig_hw
    prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    pmin, pmax = prob.min(), prob.max()
    if pmax - pmin > 1e-8:
        prob = (prob - pmin) / (pmax - pmin)
    mask = (prob > bin_thresh).astype(np.uint8) * 255
    return prob, mask

def _save_outputs(out_dir: Path, stem: str, mask: np.ndarray, prob: np.ndarray = None, overlay_src: np.ndarray = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    if prob is not None:
        cv2.imwrite(str(out_dir / f"{stem}_prob.png"), (prob * 255).astype(np.uint8))

    if overlay_src is not None:
        ov = overlay_src.copy()
        alpha = 0.45
        ov[mask > 127] = (ov[mask > 127] * (1 - alpha) + np.array([0, 0, 255]) * alpha).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), ov)

    return mask_path

def predict_torchscript(model_path, input_path, output_dir="preds",
                        bin_thresh=0.5, save_prob=False, save_overlay=False, device=None):
    """Predice usando un modelo TorchScript exportado (.pt)"""
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Cargando modelo TorchScript en {device}...")
    model = torch.jit.load(str(model_path), map_location=device).eval()

    if input_path.is_dir():
        imgs = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in EXTS]
    else:
        if input_path.suffix.lower() not in EXTS:
            raise ValueError(f"Extensión no soportada: {input_path.suffix}")
        imgs = [input_path]

    if not imgs:
        print("[INFO] No se encontraron imágenes.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Procesando {len(imgs)} imágenes...")

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] No pude leer: {p}")
            continue

        ten, orig_hw = _preprocess_bgr(img)
        ten = ten.to(device)

        with torch.no_grad():
            pred = model(ten)  # salida (1,1,H,W)
            if pred.ndim == 4:
                pred = F.interpolate(pred, size=orig_hw, mode='bilinear', align_corners=True)
            prob = pred.squeeze().detach().cpu().numpy().astype(np.float32)

        prob, mask = _postprocess_prob(prob, orig_hw, bin_thresh)
        _save_outputs(output_dir, p.stem, mask, prob if save_prob else None, img if save_overlay else None)

    print("[OK] Listo. Resultados en:", output_dir)