#  Polyp Diagnostic Tool

**Polyp Diagnostic Tool** es una aplicación de apoyo diagnóstico basada en visión artificial para la detección de pólipos en imágenes de colonoscopia.  
El software permite realizar inferencias individuales o por carpetas utilizando un modelo de segmentación entrenado con redes neuronales (UACANet / U-Net).

 *Este software es únicamente una herramienta de apoyo diagnóstico y no reemplaza el criterio médico profesional.*

---

##  Características

- Interfaz gráfica desarrollada en **Tkinter**
- Detección y segmentación automática de pólipos
- Inferencia individual o por lotes (carpeta completa)
- Visualización con superposición de resultados en rojo
- Control de umbral ajustable y filtrado por área mínima
- Modo oscuro con diseño moderno

---

##  Requisitos

- **Python 3.10+**
- **Librerías principales:**
  ```bash
  pip install torch torchvision torchaudio
  pip install opencv-python pillow numpy matplotlib

# Estructura del proyecto
PolypDiagnosticTool/
├── interface.py        # Interfaz gráfica principal
├── uacanet_ts.pt         # Carga del modelo UACANet
├── eval_ts/                  # graficas y metricas de evaluacion
├── predicts/                # Carpeta donde se guardan las inferencias
├── testDataset              # dataset para realizar pruebas y metricas
└── README.md               # Este archivo

# Descargar modelo preentrenado

El modelo .pt no se incluye aquí por su tamaño.
Puedes descargarlo desde Google Drive:

 Descargar modelo (.pt) desde Drive: https://drive.google.com/file/d/1HPtpV59-f5ZwpYPp-tIfXTWEB6zTWm5C/view?usp=drive_link

Después de descargarlo, colócalo en la raiz de la carpeta del proyecto

# Cómo ejecutar
 python interface.py
