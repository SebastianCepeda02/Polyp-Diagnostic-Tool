import predict

predict.predict_torchscript(
    "uacanet_ts.pt",   # tu modelo TorchScript
    "imagen/149.png",               # imagen a predecir
    output_dir="preds",
    save_overlay=True,
    save_prob=True
)

## pip install torch opencv-python pillow numpy
