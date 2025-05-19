# ASL Alphabet Recognition with CNN

This project recognizes **American Sign Language (ASL)** alphabets using a Convolutional Neural Network (CNN) trained on image data. It consists of two parts:

* `train_model.py`: Trains a CNN to recognize ASL letters.
* `translation.py`: Uses a webcam to recognize ASL hand signs in real-time.


## 📁 Folder Structure

project/
|
├── train_model.py         # Train and save the model
├── translation.py         # Real-time ASL prediction using webcam
├── saved_model/
│   └── asl_alphabet_model.h5
├── asl_alphabet_train/    # ASL training dataset (A-Z folders with images)


## ✅ Requirements

Install dependencies:

pip install opencv-python numpy matplotlib tensorflow scikit-learn


## 🧠 Training the Model

1. **Download the ASL dataset**:
   Use the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in the `asl_alphabet_train/` folder.

2. **Run the training script**:

python train_model.py

* Trains a CNN on images resized to `64x64`.
* Uses 1000 images per class for faster training.
* Saves the model to `saved_model/asl_alphabet_model.h5`.


## 🤖 Real-Time Translation

Once the model is trained and saved:

python translation.py

* Opens your webcam.
* Draws a green box to capture your hand sign.
* Predicts the ASL letter and shows the confidence score.

### Controls:

* Press **`q`** to quit the webcam view.


## 🔍 Notes

* The model assumes a consistent camera angle and lighting.
* The region of interest (ROI) is hardcoded (`100x100` to `300x300`) — adjust if needed.
* You can improve accuracy by training on more images or using a more complex model.

