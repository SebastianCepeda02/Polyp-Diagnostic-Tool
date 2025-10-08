# eval_uacanet_ts.py  (versión con batch_size)
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ----------------------------- Config -----------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET = (352, 352)  # H, W que espera UACANet típicamente
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --------------------------- Pre/Post -----------------------------
def preprocess_bgr(img_bgr: np.ndarray,
                   size: Tuple[int, int] = TARGET,
                   mean: np.ndarray = MEAN,
                   std: np.ndarray = STD) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    BGR -> RGB/float -> resize -> normalize -> tensor BCHW
    Devuelve: tensor (1,3,H,W), (h0,w0) originales y la imagen RGB [0..1] para visualización.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[:2]
    res = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    res = (res - mean) / std
    ten = torch.from_numpy(res.transpose(2, 0, 1)).unsqueeze(0).float()
    return ten, (h0, w0), rgb

def overlay_mask(img_rgb_01: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone una máscara binaria (0/1) en rojo sobre una imagen RGB [0..1]."""
    color = np.zeros_like(img_rgb_01); color[..., 0] = 1.0  # rojo en canal R
    mask3 = np.repeat(mask_bin[..., None], 3, axis=-1)
    return np.clip(img_rgb_01*(1-mask3) + (alpha*color + (1-alpha)*img_rgb_01)*mask3, 0, 1)

# --------------------------- Métricas -----------------------------
def confusion_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray):
    yt = y_true_bin.reshape(-1)
    yp = y_pred_bin.reshape(-1)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, tn, fp, fn

def metrics_from_counts(tp: int, tn: int, fp: int, fn: int, eps: float = 1e-8):
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    iou  = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return acc, prec, rec, iou, dice

def counts_for_thr(probs_list: List[np.ndarray], masks_list: List[np.ndarray], thr: float):
    TP=TN=FP=FN=0
    for pr, gt in zip(probs_list, masks_list):
        yp = (pr >= thr).astype(np.uint8)
        tp, tn, fp, fn = confusion_counts(gt, yp)
        TP += tp; TN += tn; FP += fp; FN += fn
    return TP, TN, FP, FN

# ------------------------ Emparejar archivos ----------------------
def load_pairs(img_dir: str, mask_dir: str, mask_suffix: Optional[str] = None):
    """
    Empareja imagen y máscara por nombre. Si mask_suffix (p.ej. '_mask'), busca 'name_mask.ext'.
    """
    img_dir_p, mask_dir_p = Path(img_dir), Path(mask_dir)
    imgs, masks, names = [], [], []

    for p in sorted(img_dir_p.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        base, ext = p.stem, p.suffix
        mname = (base + mask_suffix + ext) if mask_suffix else (base + ext)
        mp = mask_dir_p / mname
        if not mp.exists():
            # intenta encontrar por nombre base con cualquier extensión
            candidates = list(mask_dir_p.glob(base + (mask_suffix or "") + ".*"))
            if not candidates:
                continue
            mp = candidates[0]
        imgs.append(p); masks.append(mp); names.append(base)
    return imgs, masks, names

# --------------------------- Evaluación ---------------------------
@torch.no_grad()
def evaluate(model_path: str,
             img_dir: str,
             mask_dir: str,
             outdir: str = "eval_ts",
             mask_suffix: Optional[str] = None,
             device: Optional[str] = None,
             save_examples: int = 6,
             apply_sigmoid: str = "auto",  # "auto" | "yes" | "no"
             limit: Optional[int] = None,
             batch_size: int = 8) -> None:
    """
    Evalúa un modelo TorchScript (.pt) sobre un conjunto de imágenes/máscaras.
    - Soporta imágenes de distinto tamaño (no apila; calcula métricas por lista).
    - Inferencia por lotes con tamaño fijo de entrada (TARGET), salida reescalada por muestra.
    - apply_sigmoid:
        * "auto": aplica sigmoid si la salida parece logits (rango fuera de [0,1]).
        * "yes" : fuerza torch.sigmoid a la salida del modelo.
        * "no"  : asume que el modelo ya devuelve probabilidades [0,1].
    """
    outdir_p = Path(outdir)
    (outdir_p / "figs").mkdir(parents=True, exist_ok=True)
    (outdir_p / "figs" / "examples").mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Cargando TorchScript: {model_path} en {device}")
    model = torch.jit.load(str(model_path), map_location=device).eval()

    img_paths, mask_paths, names = load_pairs(img_dir, mask_dir, mask_suffix)
    if limit is not None:
        img_paths = img_paths[:limit]
        mask_paths = mask_paths[:limit]
        names = names[:limit]
    assert img_paths, "No se encontraron pares imagen-máscara"
    print(f"[INFO] Encontrados {len(img_paths)} pares")

    # Acumuladores globales
    probs_imgs: List[np.ndarray] = []
    masks_imgs: List[np.ndarray] = []
    imgs_rgb:   List[np.ndarray] = []
    ys_flat_list: List[np.ndarray] = []
    yb_flat_list: List[np.ndarray] = []

    # Buffers de lote
    batch_tens, batch_hw, batch_rgb, batch_msk = [], [], [], []

    def flush_batch():
        nonlocal batch_tens, batch_hw, batch_rgb, batch_msk
        if not batch_tens:
            return
        x = torch.cat(batch_tens, dim=0).to(device)  # (B,3,H,W)
        pred = model(x)  # (B,1,H,W) o lista/tuple
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if pred.ndim == 3:  # (B,H,W) -> (B,1,H,W)
            pred = pred.unsqueeze(1)

        # aplicar sigmoide según modo
        if apply_sigmoid == "yes":
            pred = torch.sigmoid(pred)
        elif apply_sigmoid == "auto":
            _samp = pred.detach().cpu()
            if _samp.min() < 0.0 - 1e-6 or _samp.max() > 1.0 + 1e-6:
                pred = torch.sigmoid(pred)

        # procesar cada muestra del batch
        B = pred.shape[0]
        for bi in range(B):
            ph, pw = batch_hw[bi]
            p1 = pred[bi:bi+1]  # (1,1,H,W)
            p_up = F.interpolate(p1, size=(ph, pw), mode='bilinear', align_corners=True)
            prob = p_up.squeeze().detach().cpu().numpy().astype(np.float32)  # (H,W)

            msk_r = cv2.resize(batch_msk[bi], (pw, ph), interpolation=cv2.INTER_NEAREST)
            msk_r = (msk_r > 127).astype(np.uint8)

            probs_imgs.append(prob)
            masks_imgs.append(msk_r)
            imgs_rgb.append(batch_rgb[bi])

            ys_flat_list.append(prob.reshape(-1))
            yb_flat_list.append(msk_r.reshape(-1))

        # limpiar buffers
        batch_tens, batch_hw, batch_rgb, batch_msk = [], [], [], []

    # Llenado de lotes
    for p_img, p_mask, nm in zip(img_paths, mask_paths, names):
        img_bgr = cv2.imread(str(p_img))
        msk     = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or msk is None:
            print(f"[WARN] No se pudo leer: {p_img} o {p_mask}")
            continue

        ten, orig_hw, rgb_vis = preprocess_bgr(img_bgr)  # (1,3,H,W), (h0,w0), RGB[0..1]
        batch_tens.append(ten)
        batch_hw.append(orig_hw)
        batch_rgb.append(rgb_vis)
        batch_msk.append(msk)

        if len(batch_tens) >= batch_size:
            flush_batch()

    # Último flush
    flush_batch()

    # vectores globales
    ys_all = np.concatenate(ys_flat_list)
    yb_all = np.concatenate(yb_flat_list)
    print(f"[INFO] Píxeles totales para métricas: {ys_all.shape[0]:,}")

    # ------------------ Métricas umbral-independientes ------------------
    fpr, tpr, _ = roc_curve(yb_all, ys_all)
    roc_auc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(yb_all, ys_all)
    ap = average_precision_score(yb_all, ys_all)

    plt.figure()
    plt.plot(fpr, tpr); plt.plot([0,1], [0,1], '--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(outdir_p/"figs/01_roc.png", dpi=300); plt.close()

    plt.figure()
    plt.plot(rec_c, prec_c)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision-Recall (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(outdir_p/"figs/02_pr.png", dpi=300); plt.close()

    # ---------------------- Barrido de umbral (Dice) --------------------
    ths = np.linspace(0.1, 0.9, 17)
    dices = []
    for t in ths:
        TP, TN, FP, FN = counts_for_thr(probs_imgs, masks_imgs, float(t))
        eps = 1e-8
        dice_t = (2*TP) / (2*TP + FP + FN + eps)
        dices.append(dice_t)

    best_idx = int(np.argmax(dices))
    best_thr = float(ths[best_idx])
    best_dice = float(dices[best_idx])

    plt.figure()
    plt.plot(ths, dices, marker='o'); plt.axvline(best_thr, ls='--')
    plt.xlabel("Umbral"); plt.ylabel("Dice")
    plt.title(f"Barrido de umbral (mejor thr={best_thr:.2f}, Dice={best_dice:.3f})")
    plt.tight_layout(); plt.savefig(outdir_p/"figs/03_threshold_sweep.png", dpi=300); plt.close()

    # --------------------- Métricas @ mejor umbral ----------------------
    TP, TN, FP, FN = counts_for_thr(probs_imgs, masks_imgs, best_thr)
    acc, prec, rec, iou, dice = metrics_from_counts(TP, TN, FP, FN)

    # -------------------- Métricas por imagen (avg±DE) -----------------
    dices_img, ious_img = [], []
    for pr, gt in zip(probs_imgs, masks_imgs):
        yp = (pr >= best_thr).astype(np.uint8)
        tp, tn, fp, fn = confusion_counts(gt, yp)
        *_, iou_i, dice_i = metrics_from_counts(tp, tn, fp, fn)
        dices_img.append(dice_i); ious_img.append(iou_i)
    dices_img = np.array(dices_img); ious_img = np.array(ious_img)

    plt.figure()
    plt.hist(dices_img, bins=20)
    plt.xlabel("Dice por imagen"); plt.ylabel("Frecuencia")
    plt.tight_layout(); plt.savefig(outdir_p/"figs/04_hist_dice.png", dpi=300); plt.close()

    # -------------------------- Ejemplos visuales -----------------------
    k = min(save_examples, len(probs_imgs))
    if k > 0:
        idxs = np.random.choice(len(probs_imgs), size=k, replace=False)
        for i in idxs:
            img = imgs_rgb[i]
            gt  = masks_imgs[i].astype(np.uint8)
            prb = probs_imgs[i]
            pr  = (prb >= best_thr).astype(np.uint8)
            ov  = overlay_mask(img, pr)

            # Probabilidad normalizada solo para visualizar
            pv = (prb - prb.min())/(prb.max() - prb.min() + 1e-8)

            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs[0].imshow(img); axs[0].set_title("Original"); axs[0].axis('off')
            axs[1].imshow(gt, cmap='gray'); axs[1].set_title("GT"); axs[1].axis('off')
            axs[2].imshow(pr, cmap='gray'); axs[2].set_title(f"Pred (thr={best_thr:.2f})"); axs[2].axis('off')
            axs[3].imshow(pv, cmap='inferno'); axs[3].set_title("Prob (vis)"); axs[3].axis('off')
            plt.tight_layout(); plt.savefig(outdir_p/f"figs/examples/ex_{i}.png", dpi=300); plt.close()

    # ------------------------------ Resumen -----------------------------
    resumen = f"""
Resultados con TorchScript: {Path(model_path).name}
Pares de test: {len(probs_imgs)}
Umbral-independientes:
  - ROC-AUC: {roc_auc:.4f}
  - AP (PR-AUC): {ap:.4f}
Barrido de umbral:
  - Mejor umbral (Dice): {best_thr:.2f}
Métricas pixel a pixel @thr={best_thr:.2f}:
  - Dice: {dice:.4f}
  - IoU : {iou:.4f}
  - Prec: {prec:.4f}
  - Rec : {rec:.4f}
  - Acc : {acc:.4f}
Por imagen (media ± DE):
  - Dice: {dices_img.mean():.4f} ± {dices_img.std():.4f}
  - IoU : {ious_img.mean():.4f} ± {ious_img.std():.4f}
% imágenes con Dice ≥ 0.70: {(np.mean(dices_img >= 0.70) * 100):.1f}%
Figuras: {outdir_p/'figs'}
"""
    print(resumen)
    with open(outdir_p/"00_resumen.txt", "w", encoding="utf-8") as f:
        f.write(resumen)

# ------------------------------- CLI --------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluación de UACANet TorchScript (.pt) con métricas y figuras")
    ap.add_argument("--model", required=True, help="Ruta al modelo TorchScript .pt (p.ej., uacanet_ts.pt)")
    ap.add_argument("--images", required=True, help="Carpeta de imágenes (p.ej., TestDataset/Kvasir/images)")
    ap.add_argument("--masks",  required=True, help="Carpeta de máscaras GT (p.ej., TestDataset/Kvasir/masks)")
    ap.add_argument("--out", default="eval_ts", help="Carpeta de salida (por defecto: eval_ts)")
    ap.add_argument("--mask_suffix", default=None, help="Sufijo de la máscara si aplica (p.ej., _mask)")
    ap.add_argument("--device", default=None, help="cpu | cuda (auto por defecto)")
    ap.add_argument("--examples", type=int, default=6, help="Cuántos ejemplos visuales guardar")
    ap.add_argument("--apply_sigmoid", choices=["auto","yes","no"], default="auto",
                    help="Aplicar sigmoide a la salida del modelo (auto detecta logits)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluar solo las primeras N imágenes")
    ap.add_argument("--bs", type=int, default=8, help="Batch size para inferencia")
    args = ap.parse_args()

    evaluate(model_path=args.model,
             img_dir=args.images,
             mask_dir=args.masks,
             outdir=args.out,
             mask_suffix=args.mask_suffix,
             device=args.device,
             save_examples=args.examples,
             apply_sigmoid=args.apply_sigmoid,
             limit=args.limit,
             batch_size=args.bs)
