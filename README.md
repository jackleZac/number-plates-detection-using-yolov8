# 🚗 Number Plate Detection using YOLOv8

This project implements an **Automatic Number Plate Detection System** using **YOLOv8**, trained on a custom dataset of vehicle images and license plate annotations.
It includes full data preprocessing, model training, validation, prediction, and result visualization.

---

## 📌 **Project Overview**

This project uses **Ultralytics YOLOv8** to detect number plates in real-world images.
The workflow includes:

* Dataset structure validation
* Loading and inspecting images/labels
* Configuring and training YOLOv8
* Evaluating training metrics
* Running inference on test samples
* Saving annotated predictions

This project can be extended into:

* Number plate **recognition (OCR)**
* Smart parking systems
* Access control systems
* Traffic enforcement toolkits

---

## 🗂️ **Project Structure**

```
project/
│
├── number-plate-detection-using-yolov8.ipynb       # This project's notebook
└── README.md
```

---

## 🧠 **Model Training**

You trained YOLOv8 using:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

Training outputs include:

* Precision / Recall curves
* Confusion matrix
* Loss curves
* F1 curve

All stored inside:

```
runs/detect/train/
```

---

## 🔍 **Validation + Sample Predictions**

Your notebook includes:

```python
results = model.predict(source=test_image)
results[0].save()
```

This generates bounding-box annotated outputs inside:

```
runs/detect/predict/
```

You can also manually select any test image for prediction.

---

## 🛠️ **Dependencies**

Install YOLOv8:

```bash
pip install ultralytics
```

Other dependencies:

```bash
pip install numpy matplotlib opencv-python
```

---

## 🎯 **Features**

* Custom YOLOv8 training
* Fast inference on unseen images
* Automatic saving of prediction outputs
* Dataset integrity checks
* Clean visualization of results

---

## 🚀 **How to Run**

1. Clone the repository
2. Ensure the dataset is linked in `dataset.yaml`
3. Train the model:

```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

4. Predict:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
```

---

## 📦 **Future Improvements**

* Add license plate **OCR recognition**
* Convert into a **Streamlit or FastAPI** web app
* Support for multiple regions/countries
* Real-time inference on CCTV footage

---

## 🏆 **Author**
https://github.com/jackleZac

## Note
Download the following dataset from Kaggle: https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset
