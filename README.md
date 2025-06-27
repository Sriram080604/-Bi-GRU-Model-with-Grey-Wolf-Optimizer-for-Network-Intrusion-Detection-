# -Bi-GRU-Model-with-Grey-Wolf-Optimizer-for-Network-Intrusion-Detection-
This project implements a **Bidirectional GRU (Bi-GRU)** deep learning model optimized using the **Grey Wolf Optimizer (GWO)** to detect network-based attacks from flow-based features. The model is trained and evaluated on a labeled dataset to achieve high accuracy in classifying attack types.

---

## 📌 Abstract

Network intrusion detection is critical for cybersecurity in modern IT infrastructures. This project combines the power of **deep learning (Bi-GRU)** and **metaheuristic optimization (GWO)** to develop a robust intrusion detection system (IDS). GWO is used to tune the model’s hyperparameters, improving the overall detection accuracy and minimizing false positives.

---

## 🧰 Tools & Technologies Used

- 🐍 Python 3.x
- 🧠 TensorFlow / Keras
- 🔄 NumPy, Pandas
- 📊 Matplotlib, Seaborn
- 🧪 Scikit-learn
- 🐺 Grey Wolf Optimizer (GWO)
- 📁 Dataset: KDD-based network flow dataset (6-class classification)

---

## ⚙️ Steps Involved

1. **Data Preprocessing**  
   - Cleaned and normalized flow-based features  
   - Encoded labels for 6 attack classes

2. **Model Building**  
   - Constructed a Bi-GRU model using Keras  
   - Tuned GRU units and dropout rate using GWO

3. **Optimization**  
   - Used GWO to optimize the model’s hyperparameters  
   - Fitness function based on validation loss

4. **Training and Evaluation**  
   - Model trained for 10 epochs  
   - Achieved 100% accuracy and perfect classification on the validation set  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Cohen’s Kappa

5. **Visualization**  
   - Plotted fitness curve, accuracy/loss graphs  
   - Confusion matrix and ROC curves for each class

---

## 📈 Results

| Metric            | Value |
|-------------------|-------|
| Accuracy          | 1.00  |
| Precision         | 1.00  |
| Recall            | 1.00  |
| F1-Score          | 1.00  |
| Cohen’s Kappa     | 1.00  |

All six classes were predicted correctly with zero misclassification.

---

## 📌 Visual Outputs

- Fitness Curve Before Tuning  
- Accuracy & Loss Graphs  
- Confusion Matrix  
- ROC Curve (AUC = 1.0 for all classes)

---

## ✅ Conclusion

The project demonstrates the effectiveness of using **Grey Wolf Optimizer** for fine-tuning Bi-GRU models in the context of **network attack detection**. The final model achieves **state-of-the-art performance**, making it highly suitable for real-time cybersecurity systems.

---
