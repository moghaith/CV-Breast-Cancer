# 🧠 Breast Ultrasound Cancer Classification (BUSI Dataset)

This project implements **deep learning models** for classifying breast ultrasound images into:

- **Benign**
- **Malignant**
- **Normal**

It uses **ResNet18** and **MobileNetV2** architectures with different preprocessing techniques and evaluates performance through confusion matrices. It also includes deployment via **Gradio + Hugging Face Spaces**.

---

## 📂 Dataset

- Dataset used: [Dataset_BUSI_with_GT](https://www.kaggle.com/datasets/aysendegerli/breast-ultrasound-images-dataset)
- The original dataset is placed in:  
  ```
  Dataset_BUSI_with_GT/
  ```
- The dataset is split into training and validation:
  ```
  BUSI_split/
  ```

---

## 🧠 Models

### ✅ Implemented Architectures:
- `ResNet18`
- `MobileNetV2`

### 📊 Trained Variants:
- Raw images
- Z-score normalized
- Non-Local Means Denoised (NLMD)
- Preprocessed (custom preprocessing)

### 🗃️ Model Weights:
Model weights are saved in `.pth` files:

- `resnet18_busi.pth`
- `resnet18_busi_zscore.pth`
- `mobilenetv2_busi_nlmd.pth`
- (and more)

---

## 🧪 Confusion Matrices

Confusion matrix images are included for analysis:

- `confusion_matrix_resnet18_preprocessed.png`
- `confusion_matrix_mobilenet_preprocessed.png`
- ...and others

These provide insights into model accuracy and class-wise performance.

---

## 📦 Code Files

- `CNN_code.ipynb`: Jupyter notebook with training and evaluation logic.
- `train.py` *(if separated)*: Contains training script logic.
- `preprocessing/`: Scripts for image preprocessing (z-score, NLMD, etc.)
- `app.py`: Gradio app for live inference.
- `requirements.txt`: Required libraries for deployment.

---

## 🚀 Deployment with Hugging Face Spaces

A `Gradio` UI is created for real-time classification of ultrasound images.

### 🧾 Files:
- `app.py`: Gradio interface logic
- `resnet18_busi.pth`: Trained ResNet18 model
- `requirements.txt`: Lists `torch`, `torchvision`, `gradio`, and more

### ✅ Steps to Deploy:
1. Go to: https://huggingface.co/spaces
2. Create a new Space → Choose **Gradio**
3. Upload:
   - `app.py`
   - `resnet18_busi.pth`
   - `requirements.txt`
4. Wait for the build → You’ll get a live demo page.

#### Example interface:
```python
gr.Interface(fn=classify_image, 
             inputs=gr.Image(type="pil"), 
             outputs=gr.Label(num_top_classes=3)).launch()
```

🔗 [View Live Space](https://huggingface.co/spaces/moghaith/breastcancerclassify)

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open your browser at `http://127.0.0.1:7860/`

---

## 📊 Example Output

```json
{
  "benign": 0.03,
  "malignant": 0.95,
  "normal": 0.02
}
```

---

## 📚 References

- [BUSI Dataset on Kaggle](https://www.kaggle.com/datasets/aysendegerli/breast-ultrasound-images-dataset)
- [Gradio Docs](https://gradio.app/)
- [PyTorch Models](https://pytorch.org/vision/stable/models.html)

---

## 👨‍💻 Author

**Mohamed Ghaith**  
🔗 GitHub: [moghaith](https://github.com/moghaith)
