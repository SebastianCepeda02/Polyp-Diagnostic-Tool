# run_eval_and_predict.py
import predict
import eval_uacanet_ts as evalts  # asegúrate que el archivo se llame así y exponga evaluate(...)

MODEL = "uacanet_ts.pt"
IMAGES = "TestDataset/Kvasir/images"
MASKS  = "TestDataset/Kvasir/masks"

# 1) algunas predicciones rápidas
#predict.predict_torchscript(MODEL, IMAGES, output_dir="preds", save_overlay=True, save_prob=True)

# 2) evaluación con métricas + figuras
evalts.evaluate(MODEL, IMAGES, MASKS, outdir="eval_ts", mask_suffix=None, batch_size=8, device=None, save_examples=6)
