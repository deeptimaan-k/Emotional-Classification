# 😃 EmoSense – Real-Time Emotion Detection System  
### by [@deeptimaan-k](https://github.com/deeptimaan-k)

> 🎯 *An AI-powered emotion recognition system that detects human emotions in real-time using Deep Learning and Computer Vision — wrapped in a modern Tkinter GUI.*

---

## 🧠 Overview

**EmoSense** is a **real-time facial emotion recognition system** built with **TensorFlow, Keras, OpenCV**, and **Tkinter**.  
It captures live webcam video, detects faces, predicts the user’s emotion, and displays both **emoji reactions** and **motivational messages** dynamically.

The model is trained on grayscale facial expression data and predicts **7 basic human emotions** with impressive accuracy.

---

## 💡 Features

✅ **Real-time Emotion Detection** using webcam feed  
✅ **Deep Learning CNN model** trained on facial expression dataset  
✅ **Live Face Detection** using Haar Cascade  
✅ **Beautiful Tkinter UI** with dual panels (video + emotion info)  
✅ **Emoji & Reaction system** for engaging feedback  
✅ **Confidence score** display for every prediction  
✅ **Cross-platform support** (Windows, macOS, Linux)  
✅ **Lightweight, Offline, and Fast**

---

## 🧩 Supported Emotions

| Emotion | Emoji | Description |
|:--|:--:|:--|
| Angry 😠 | 🔥 | Take a deep breath and relax! |
| Disgusted 🤢 | 😣 | Something’s bothering you? |
| Fearful 😨 | 😱 | Don’t worry, everything will be fine! |
| Happy 😀 | 😃 | Keep smiling! You’re doing great! |
| Neutral 😐 | 😶 | You seem calm and composed. |
| Sad 😔 | 😞 | Cheer up! Better days are ahead. |
| Surprised 😲 | 😮 | Wow! That’s surprising! |

---

## 🧱 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Programming Language** | Python 3.x |
| **Libraries (AI/ML)** | TensorFlow, Keras, NumPy |
| **Computer Vision** | OpenCV |
| **GUI Development** | Tkinter, PIL (Pillow) |
| **Data Augmentation** | ImageDataGenerator |
| **Model Architecture** | CNN (Convolutional Neural Network) |

---

## 🏗️ Project Architecture

```
📦 EmoSense
├── data/
│   ├── train/        # Training images (7 folders for each emotion)
│   ├── test/         # Validation images
│
├── emojis/           # Emoji icons for each emotion
│   ├── angry.png
│   ├── happy.png
│   ├── sad.png
│   └── ...etc
│
├── emotion_model.h5             # Saved trained model
├── emotion_model.weights.h5     # Model weights
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── main.py                      # Tkinter GUI and real-time detection
├── train_model.py               # Model training and saving script
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/deeptimaan-k/EmoSense.git
cd EmoSense
```

### 2️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow
opencv-python
numpy
pillow
tk
```

### 3️⃣ (Optional) Train Your Own Model
If you want to train the model again:
```bash
python train_model.py
```

Make sure your dataset is organized as:
```
data/
  ├── train/
  │   ├── angry/
  │   ├── happy/
  │   ├── ...
  ├── test/
      ├── angry/
      ├── happy/
      ├── ...
```

### 4️⃣ Run the Application
```bash
python main.py
```

---

## 🧑‍💻 Model Details

| Layer Type | Filters | Kernel Size | Activation | Notes |
|-------------|----------|--------------|-------------|-------|
| Conv2D | 32 | (3x3) | ReLU | Input layer |
| Conv2D | 64 | (3x3) | ReLU | Feature extraction |
| MaxPooling2D | — | (2x2) | — | Downsampling |
| Dropout | 0.25 | — | — | Regularization |
| Conv2D | 128 | (3x3) | ReLU | Deep features |
| Conv2D | 128 | (3x3) | ReLU | Deep features |
| MaxPooling2D | — | (2x2) | — | Downsampling |
| Dropout | 0.25 | — | — | Regularization |
| Flatten | — | — | — | Vectorization |
| Dense | 1024 | — | ReLU | Fully connected |
| Dropout | 0.5 | — | — | Regularization |
| Dense | 7 | — | Softmax | Output (7 classes) |

**Loss:** Categorical Crossentropy  
**Optimizer:** Adam (lr=0.0001, decay=1e-6)  
**Accuracy:** ~85–90% (depending on dataset)

---

## 🖥️ GUI Preview

🪄 **Left Panel:** Real-time webcam feed with face bounding boxes  
💬 **Right Panel:** Detected emotion, emoji, confidence level, and reaction message  

```
 -----------------------------------------------------------
|  [ Webcam Feed ]                |  Emotion: Happy 😀     |
|                                 |  Confidence: 97.5%     |
|                                 |  Keep smiling! 😀       |
|                                 |  [ Emoji Display ]     |
 -----------------------------------------------------------
```

---

## 🧪 Example Output

**Input (webcam frame):**
> Face detected smiling  

**Predicted Output:**
```
Emotion: Happy 😀
Confidence: 96.3%
Reaction: "Keep smiling! You're doing great!"
```

---

## 📸 Emojis Folder Example

```
emojis/
 ├── angry.png
 ├── disgusted.png
 ├── fearful.png
 ├── happy.png
 ├── neutral.png
 ├── sad.png
 └── surprised.png
```

---

## 🛡️ Error Handling & Features

- Handles **camera access errors** gracefully  
- Displays **“No Face Detected”** message dynamically  
- **Auto-smooths predictions** over frames to reduce jitter  
- **Responsive GUI layout** centered on screen  
- **OS detection** for better webcam performance on macOS  

---

## 🌍 Cross-Platform Support

| OS | Supported | Tested |
|----|------------|---------|
| 🪟 Windows 10/11 | ✅ | ✅ |
| 🍎 macOS | ✅ | ✅ |
| 🐧 Linux | ✅ | ✅ |

---

## 🚀 Future Enhancements

🔹 Emotion-based music or mood recommendations  
🔹 Voice feedback using text-to-speech  
🔹 Cloud-based emotion analytics dashboard  
🔹 Integration with smart home or IoT systems  
🔹 Mobile app version using React Native + TensorFlow Lite

---

## 🧾 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## 💖 Credits

Developed with ❤️ by **[deeptimaan-k](https://github.com/deeptimaan-k)**  
> “Emotions speak louder than words — now, your computer understands them too.” 🧠💫

---

## ⭐ Support

If you like this project, please consider **starring ⭐ the repo** on GitHub.  
Your support helps improve and inspire more AI-powered innovations!
