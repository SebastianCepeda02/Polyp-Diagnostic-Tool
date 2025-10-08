from pathlib import Path

# Ruta por defecto al modelo TorchScript (junto al .py)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "uacanet_ts.pt"


import math
import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F

# ---- Config de preprocesado (mismo que en el wrapper) ----
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET = (352, 352)  # (H, W) del modelo exportado

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def preprocess_bgr(img_bgr, size=TARGET, mean=MEAN, std=STD):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[:2]
    res = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    res = (res - mean) / std
    ten = torch.from_numpy(res.transpose(2,0,1)).unsqueeze(0).float()
    return ten, (h0, w0)

def postprocess_prob(prob, orig_hw):
    # prob puede venir ya en 0..1; igual normalizamos por seguridad
    h0, w0 = orig_hw
    prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    pmin, pmax = float(prob.min()), float(prob.max())
    if pmax - pmin > 1e-8:
        prob = (prob - pmin) / (pmax - pmin)
    else:
        prob = np.zeros_like(prob, dtype=np.float32)
    return prob

def to_overlay(bgr, mask_bin, alpha=0.45):
    ov = bgr.copy()
    # rojo en BGR (0,0,255)
    ov[mask_bin > 127] = (ov[mask_bin > 127] * (1 - alpha) + np.array([0,0,255]) * alpha).astype(np.uint8)
    return ov

class UACANetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Pólipos - Software de Apoyo Diagnóstico")

        # Estado
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_path = None
        self.img_bgr = None
        self.img_path = None
        self.prob = None
        self.mask_bin = None

        self.mm_per_px = None  # calibración opcional (mm por píxel)
        self.draw_all = tk.BooleanVar(value=False)  # dibujar todos los componentes o sólo el mayor

        self.min_area_px = 100  # área mínima para considerar un pólipo

        # UI
        self.build_ui()

        # 🔹 Carga automática del modelo por defecto
        try:
            self.info("Cargando modelo por defecto...")
            self.model = torch.jit.load(str(DEFAULT_MODEL_PATH), map_location=self.device).eval()
            self.model_path = str(DEFAULT_MODEL_PATH)
            # habilitar botones que dependen del modelo
            self.btn_load_img.config(state=tk.NORMAL)
            self.btn_batch.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.DISABLED)
            self.btn_save_mask.config(state=tk.DISABLED)
            self.btn_save_overlay.config(state=tk.DISABLED)
            self.info(f"Modelo cargado: {Path(self.model_path).name} | Dispositivo: {self.device}")
        except Exception as e:
            self.info("No se pudo cargar el modelo por defecto.")
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se pudo cargar el modelo por defecto en:\n{DEFAULT_MODEL_PATH}\n\nError:\n{e}"
            )

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        #self.btn_load_model = ttk.Button(top, text="Cargar modelo (.pt)", command=self.load_model_dialog)
        #self.btn_load_model.pack(side=tk.LEFT, padx=4)

        #self.lbl_device = ttk.Label(top, text=f"Dispositivo: {self.device}")
        #self.lbl_device.pack(side=tk.LEFT, padx=12)

        self.btn_load_img = ttk.Button(top, text="Abrir imagen", command=self.load_image_dialog, state=tk.DISABLED)
        self.btn_load_img.pack(side=tk.LEFT, padx=4)

        self.btn_batch = ttk.Button(top, text="Predicción por carpeta", command=self.batch_dialog, state=tk.DISABLED)
        self.btn_batch.pack(side=tk.LEFT, padx=4)

        self.th_label = ttk.Label(top, text="Umbral:")
        self.th_label.pack(side=tk.LEFT, padx=(16,2))
        self.th_var = tk.DoubleVar(value=0.50)
        self.th_scale = ttk.Scale(top, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.th_var, command=self.on_threshold_change, length=150)
        self.th_scale.pack(side=tk.LEFT)

        # mm/px entry
        self.mmpx_label = ttk.Label(top, text="mm/px:")
        self.mmpx_label.pack(side=tk.LEFT, padx=(16, 2))
        self.mmpx_var = tk.StringVar(value="")
        self.mmpx_entry = ttk.Entry(top, width=7, textvariable=self.mmpx_var)
        self.mmpx_entry.pack(side=tk.LEFT)
        ttk.Button(top, text="Aplicar", command=self.apply_mmpx).pack(side=tk.LEFT, padx=4)

        # checkbox: dibujar todos los componentes
        ttk.Checkbutton(top, text="Todos los componentes", variable=self.draw_all, command=self.refresh_display).pack(
            side=tk.LEFT, padx=(12, 4))

        self.btn_predict = ttk.Button(top, text="Predecir", command=self.predict_once, state=tk.DISABLED)
        self.btn_predict.pack(side=tk.LEFT, padx=8)

        self.btn_save_mask = ttk.Button(top, text="Guardar máscara", command=self.save_mask, state=tk.DISABLED)
        self.btn_save_mask.pack(side=tk.LEFT, padx=4)

        self.btn_save_overlay = ttk.Button(top, text="Guardar overlay", command=self.save_overlay, state=tk.DISABLED)
        self.btn_save_overlay.pack(side=tk.LEFT, padx=4)

        # Área de imagen
        self.canvas = tk.Canvas(self.root, bg="#222", width=1200, height=600, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<Configure>", lambda e: self.refresh_display())

        # Barra de estado
        self.status = tk.StringVar(value="Cargue el modelo (.pt)")
        self.status_bar = ttk.Label(self.root, textvariable=self.status, anchor="w", padding=(8,4))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def info(self, msg): self.status.set(msg); self.root.update_idletasks()

    def load_model_dialog(self):
        path = filedialog.askopenfilename(title="Selecciona el modelo TorchScript (.pt)", filetypes=[("TorchScript", "*.pt *.pth")])
        if not path:
            return
        try:
            self.info("Cargando modelo...")
            self.model = torch.jit.load(path, map_location=self.device).eval()
            self.model_path = path
            self.btn_load_img.config(state=tk.NORMAL)
            self.btn_batch.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.DISABLED)
            self.btn_save_mask.config(state=tk.DISABLED)
            self.btn_save_overlay.config(state=tk.DISABLED)
            self.info(f"Modelo cargado: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
            self.info("Error al cargar el modelo.")

    def load_image_dialog(self):
        path = filedialog.askopenfilename(title="Selecciona una imagen",
                                          filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", "No pude leer la imagen.")
            return
        self.img_bgr = bgr
        self.img_path = path
        self.prob = None
        self.mask_bin = None
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_save_mask.config(state=tk.DISABLED)
        self.btn_save_overlay.config(state=tk.DISABLED)
        self.info(f"Imagen cargada: {os.path.basename(path)}")
        self.refresh_display()

    def predict_once(self):
        if self.model is None or self.img_bgr is None:
            return
        self.info("Inferencia...")
        self.btn_predict.config(state=tk.DISABLED)
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        try:
            ten, orig_hw = preprocess_bgr(self.img_bgr)
            ten = ten.to(self.device)
            with torch.no_grad():
                pred = self.model(ten)  # (1,1,H,W), TorchScript con sigmoid adentro
                if pred.ndim == 4:
                    pred = F.interpolate(pred, size=orig_hw, mode='bilinear', align_corners=True)
                prob = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            # Normaliza por seguridad (0..1)
            self.prob = postprocess_prob(prob, orig_hw)
            self.update_mask_from_threshold()
            #self.info("Listo.")
        except Exception as e:
            messagebox.showerror("Error", f"Fallo en inferencia:\n{e}")
            self.info("Error en inferencia.")
        finally:
            self.btn_predict.config(state=tk.NORMAL)

    def on_threshold_change(self, _evt=None):
        if self.prob is not None:
            self.update_mask_from_threshold()

    def update_mask_from_threshold(self):
        th = float(self.th_var.get())
        mask = (self.prob > th).astype(np.uint8) * 255

        # 🔎 filtrar componentes pequeños en la máscara
        mask = self._filter_small_components(mask, self.min_area_px)

        self.mask_bin = mask
        self.btn_save_mask.config(state=tk.NORMAL)
        self.btn_save_overlay.config(state=tk.NORMAL)

        self.refresh_display()

        if self.mask_bin is not None:
            comps = [c for c in self._find_components(self.mask_bin) if c["area"] >= self.min_area_px]
            if comps:
                n = len(comps)
                c_max = max(comps, key=lambda cc: cc["area"])
                area_px = c_max["area"]
                if self.mm_per_px:
                    area_mm2 = area_px * (self.mm_per_px ** 2)
                    self.info(f"Pólipos detectados: {n} | Área mayor: {area_px} px ({area_mm2:.1f} mm²)")
                else:
                    self.info(f"Pólipos detectados: {n} | Área mayor: {area_px} px")
            else:
                self.info("Pólipos detectados: 0")
        else:
            self.info("Sin máscara")

    def refresh_display(self):
        # Muestra 3 paneles: original | máscara | overlay
        self.canvas.delete("all")
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W <= 10 or H <= 10:
            return

        panels = []
        titles = []

        if self.img_bgr is not None:
            img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            panels.append(Image.fromarray(img_rgb))
            titles.append("Original")

        if self.mask_bin is not None:
            # asegúrate de mostrar la máscara como imagen de 3 canales para homogeneidad
            panels.append(Image.fromarray(self.mask_bin))
            titles.append("Máscara")

            if self.img_bgr is not None:
                # overlay base
                overlay = to_overlay(self.img_bgr, self.mask_bin, alpha=0.45)

                # Filtrar componentes pequeños por el umbral unificado
                comps = [c for c in self._find_components(self.mask_bin) if c["area"] >= self.min_area_px]

                if comps:
                    if not self.draw_all.get():
                        # sólo el componente mayor por área
                        comps = [max(comps, key=lambda c: c["area"])]
                    overlay = self._annotate_boxes(overlay, comps)

                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                panels.append(Image.fromarray(overlay_rgb))
                titles.append("Overlay + bbox")

        if not panels:
            return

        cols = len(panels)
        cell_w = max(1, W // cols - 8)
        cell_h = H - 40  # ↑ da más espacio para el título

        self.tk_imgs = []
        for i, (img, title) in enumerate(zip(panels, titles)):
            img_disp = self._fit_to_box(img, (cell_w, cell_h))
            tkimg = ImageTk.PhotoImage(img_disp)
            self.tk_imgs.append(tkimg)  # mantener referencia

            x = i * (cell_w + 8) + 4
            y_title = 5
            y_img = 25  # ↓ baja la imagen para no tapar el título

            # título primero, anclaje arriba-izquierda
            self.canvas.create_text(x + 5, y_title, text=title, fill="white",
                                    anchor="nw", font=("Segoe UI", 10, "bold"))
            # imagen debajo del título
            self.canvas.create_image(x, y_img, image=tkimg, anchor="nw")

            
    def _filter_small_components(self, mask_bin: np.ndarray, min_area_px: int) -> np.ndarray:
        """Conserva solo componentes con área >= min_area_px."""
        # binaria 0/255 -> 0/1
        bw = (mask_bin > 127).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        out = np.zeros_like(bw, dtype=np.uint8)
        for lab in range(1, num):  # 0 es fondo
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                out[labels == lab] = 1
        return (out * 255).astype(np.uint8)


    def _fit_to_box(self, pil_img, box_wh):
        bw, bh = box_wh
        iw, ih = pil_img.size
        scale = min(bw / iw, bh / ih)
        if scale <= 0:
            scale = 1.0
        new_size = (max(1, int(iw*scale)), max(1, int(ih*scale)))
        return pil_img.resize(new_size, Image.BILINEAR)

    def save_mask(self):
        if self.mask_bin is None or self.img_path is None:
            return
        stem = Path(self.img_path).stem
        out = filedialog.asksaveasfilename(defaultextension=".png",
                                           initialfile=f"{stem}_mask.png",
                                           filetypes=[("PNG", "*.png")])
        if not out: return
        cv2.imwrite(out, self.mask_bin)
        self.info(f"Máscara guardada: {out}")

    def save_overlay(self):
        if self.mask_bin is None or self.img_bgr is None or self.img_path is None:
            return
        stem = Path(self.img_path).stem
        out = filedialog.asksaveasfilename(defaultextension=".png",
                                           initialfile=f"{stem}_overlay.png",
                                           filetypes=[("PNG", "*.png")])
        if not out: return
        ov = to_overlay(self.img_bgr, self.mask_bin, alpha=0.45)
        cv2.imwrite(out, ov)
        self.info(f"Overlay guardado: {out}")

    def batch_dialog(self):
        if self.model is None:
            return
        folder = filedialog.askdirectory(title="Selecciona carpeta con imágenes")
        if not folder:
            return
        out_dir = filedialog.askdirectory(title="Selecciona carpeta de salida (se crearán *_mask.png)")
        if not out_dir:
            return
        self.info("Procesando carpeta...")
        threading.Thread(target=self._batch_worker, args=(folder, out_dir), daemon=True).start()

    def _batch_worker(self, in_dir, out_dir):
        try:
            in_dir = Path(in_dir)
            out_dir = Path(out_dir)
            paths = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in EXTS]
            if not paths:
                self.info("No se encontraron imágenes en la carpeta.")
                return
            for p in paths:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                ten, orig_hw = preprocess_bgr(bgr)
                ten = ten.to(self.device)
                with torch.no_grad():
                    pred = self.model(ten)
                    if pred.ndim == 4:
                        pred = F.interpolate(pred, size=orig_hw, mode='bilinear', align_corners=True)
                    prob = pred.squeeze().detach().cpu().numpy().astype(np.float32)
                prob = postprocess_prob(prob, orig_hw)
                mask = (prob > float(self.th_var.get())).astype(np.uint8) * 255
                out_path = out_dir / f"{p.stem}_mask.png"
                cv2.imwrite(str(out_path), mask)
            self.info(f"Batch listo. Guardado en: {out_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Fallo en batch:\n{e}")
            self.info("Error en batch.")

    def apply_mmpx(self):
        txt = self.mmpx_var.get().strip()
        try:
            self.mm_per_px = float(txt) if txt else None
            if self.mm_per_px is not None and self.mm_per_px <= 0:
                raise ValueError
            self.info(f"mm/px = {self.mm_per_px if self.mm_per_px else 'no definido'}")
            self.refresh_display()
        except:
            self.mm_per_px = None
            self.info("mm/px inválido. Deja vacío o ingresa un número > 0.")

    def _find_components(self, mask_bin):
        # retorna lista de dicts con bbox y métricas por componente
        num, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_bin > 127).astype(np.uint8),
                                                                         connectivity=8)
        comps = []
        # stats: [label, x, y, w, h, area]
        for lab in range(1, num):  # 0 es fondo
            x, y, w, h, area = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP], \
                stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT], \
                stats[lab, cv2.CC_STAT_AREA]
            comps.append({"x": x, "y": y, "w": w, "h": h, "area": int(area)})
        return comps

    def _annotate_boxes(self, bgr, comps):
        out = bgr.copy()
        H, W = out.shape[:2]
        area_img = H * W
        mmpx = self.mm_per_px

        def label_text(c):
            area_px = c["area"]
            pct = 100.0 * area_px / max(1, area_img)
            if mmpx:
                area_mm2 = (area_px * (mmpx ** 2))
                # diámetro equivalente (círculo de la misma área): d = 2*sqrt(area/pi)
                d_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
                #return f"{area_px}px ({pct:.2f}%) | {area_mm2:.1f} mm² | d~{d_mm:.1f} mm"
                return f"{area_px}px | {area_mm2:.1f} mm2"
            else:
                return f"{area_px}px ({pct:.2f}%)"

        color = (0, 255, 0)
        for c in comps:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            txt = label_text(c)
            #cv2.rectangle(out, (x, max(0, y - 22)), (x + min(380, w + 200), y), color, -1)
            #cv2.putText(out, txt, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # calcular ancho del texto
            (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rect_end_x = x + text_w + 8  # margen de 8 px
            rect_end_y = y
            rect_start_y = max(0, y - text_h - 6)

            # dibujar fondo verde ajustado al texto
            cv2.rectangle(out, (x, rect_start_y), (rect_end_x, rect_end_y), color, -1)

            # escribir el texto encima
            cv2.putText(out, txt, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return out


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass
    app = UACANetGUI(root)
    root.geometry("1200x700+100+60")
    root.mainloop()

if __name__ == "__main__":
    main()
