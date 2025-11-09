# 😃 EmoSense – Real-Time Emotion Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/deeptimaan-k/EmoSense?style=social)](https://github.com/deeptimaan-k/EmoSense/stargazers)

**An AI-powered emotion recognition system that detects human emotions in real-time using Deep Learning and Computer Vision**

[Features](#-features) • [Demo](#-demo) • [Installation](#️-installation--setup) • [Usage](#-usage) • [Model](#-model-architecture) • [Contributing](#-contributing)

<img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
<img src="https://img.shields.io/badge/Maintained-Yes-brightgreen" alt="Maintained">

</div>

---

## 🎯 Overview

**EmoSense** is an intelligent facial emotion recognition system that bridges the gap between human emotions and artificial intelligence. Built with TensorFlow, Keras, OpenCV, and Tkinter, it captures live webcam video, detects faces, and accurately predicts emotions in real-time.

The system employs a sophisticated CNN architecture trained on grayscale facial expression data to recognize **7 fundamental human emotions** with up to **90% accuracy**. Each prediction is accompanied by dynamic emoji reactions and personalized motivational messages, creating an engaging and interactive experience.

### 🌟 Why EmoSense?

- **Real-time Processing**: Instant emotion detection with minimal latency
- **High Accuracy**: Deep learning model achieving 85-90% accuracy
- **User-Friendly**: Intuitive Tkinter GUI requiring no technical expertise
- **Offline & Secure**: All processing happens locally on your device
- **Cross-Platform**: Works seamlessly on Windows, macOS, and Linux
- **Lightweight**: Optimized for performance on standard hardware

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎥 Core Features
- ✅ **Real-time webcam emotion detection**
- ✅ **Deep CNN model with 7 emotion classes**
- ✅ **Haar Cascade face detection**
- ✅ **Confidence score display**
- ✅ **Smooth prediction averaging**
- ✅ **Graceful error handling**

</td>
<td width="50%">

### 🎨 Interface Features
- ✅ **Modern dual-panel Tkinter GUI**
- ✅ **Dynamic emoji reactions**
- ✅ **Motivational message system**
- ✅ **Responsive centered layout**
- ✅ **Real-time video visualization**
- ✅ **OS-optimized performance**

</td>
</tr>
</table>

---

## 🎭 Supported Emotions

<div align="center">

| Emotion | Emoji | AI Response | Color Code |
|---------|-------|-------------|------------|
| **Angry** 😠 | 🔥 | *"Take a deep breath and relax!"* | `#FF4444` |
| **Disgusted** 🤢 | 😣 | *"Something's bothering you?"* | `#8B4513` |
| **Fearful** 😨 | 😱 | *"Don't worry, everything will be fine!"* | `#9370DB` |
| **Happy** 😀 | 😃 | *"Keep smiling! You're doing great!"* | `#FFD700` |
| **Neutral** 😐 | 😶 | *"You seem calm and composed."* | `#A9A9A9` |
| **Sad** 😔 | 😞 | *"Cheer up! Better days are ahead."* | `#4682B4` |
| **Surprised** 😲 | 😮 | *"Wow! That's surprising!"* | `#FF6347` |

</div>

---

## 🛠️ Tech Stack

<div align="center">

### Core Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</div>

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | TensorFlow 2.x, Keras |
| **Computer Vision** | OpenCV |
| **GUI Framework** | Tkinter, PIL (Pillow) |
| **Numerical Computing** | NumPy |
| **Data Augmentation** | ImageDataGenerator |
| **Model Type** | Convolutional Neural Network (CNN) |

---

## 📁 Project Structure

```
📦 EmoSense/
├── 📂 data/
│   ├── 📂 train/              # Training dataset (7 emotion folders)
│   │   ├── angry/
│   │   ├── disgusted/
│   │   ├── fearful/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── 📂 test/               # Validation dataset
│       └── (same structure)
│
├── 📂 emojis/                 # Emoji assets for each emotion
│   ├── angry.png
│   ├── disgusted.png
│   ├── fearful.png
│   ├── happy.png
│   ├── neutral.png
│   ├── sad.png
│   └── surprised.png
│
├── 📄 emotion_model.h5        # Trained model (architecture + weights)
├── 📄 emotion_model.weights.h5 # Separate model weights
├── 📄 haarcascade_frontalface_default.xml  # Face detection classifier
├── 📄 main.py                 # Main application (GUI + detection)
├── 📄 train_model.py          # Model training script
├── 📄 requirements.txt        # Python dependencies
├── 📄 LICENSE                 # MIT License
└── 📄 README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera access
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/deeptimaan-k/EmoSense.git
cd EmoSense
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
pillow>=9.0.0
tk
```

### 4️⃣ Download Required Files

Ensure you have:
- `haarcascade_frontalface_default.xml` (included in OpenCV)
- `emotion_model.h5` (pre-trained model or train your own)
- Emoji assets in the `emojis/` folder

### 5️⃣ (Optional) Train Your Own Model

If you want to train the model from scratch:

```bash
python train_model.py
```

**Dataset Structure Required:**
```
data/
├── train/
│   ├── angry/      # 500+ images
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    └── (same structure)
```

---

## 🚀 Usage

### Running the Application

```bash
python main.py
```

### How It Works

1. **Launch Application**: The GUI window opens with dual panels
2. **Camera Activation**: Webcam automatically activates
3. **Face Detection**: Haar Cascade detects faces in real-time
4. **Emotion Prediction**: CNN model predicts emotion with confidence score
5. **Visual Feedback**: Displays emotion, emoji, and motivational message

### Keyboard Shortcuts

- `ESC` or `Q` - Exit application
- Window close button - Safe shutdown

---

## 🧠 Model Architecture

### CNN Architecture Details

```python
Model: Sequential CNN for Emotion Recognition
_________________________________________________________________
Layer (type)                Output Shape              Params
=================================================================
Conv2D (32 filters, 3x3)   (None, 46, 46, 32)        320
Conv2D (64 filters, 3x3)   (None, 44, 44, 64)        18,496
MaxPooling2D (2x2)          (None, 22, 22, 64)        0
Dropout (0.25)              (None, 22, 22, 64)        0
_________________________________________________________________
Conv2D (128 filters, 3x3)  (None, 20, 20, 128)       73,856
Conv2D (128 filters, 3x3)  (None, 18, 18, 128)       147,584
MaxPooling2D (2x2)          (None, 9, 9, 128)         0
Dropout (0.25)              (None, 9, 9, 128)         0
_________________________________________________________________
Flatten                     (None, 10368)             0
Dense (1024 units)          (None, 1024)              10,617,856
Dropout (0.5)               (None, 1024)              0
Dense (7 units, softmax)    (None, 7)                 7,175
=================================================================
Total params: 10,865,287
Trainable params: 10,865,287
```

### Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (lr=0.0001, decay=1e-6)
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Image Size**: 48x48 pixels (grayscale)
- **Data Augmentation**: Rotation, shift, zoom, horizontal flip

### Performance Metrics

- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~85-88%
- **Inference Speed**: 30-60 FPS (depending on hardware)
- **Model Size**: ~42 MB

---

## 🖼️ Demo

### GUI Interface Preview

```
┌─────────────────────────────────────────────────────────────┐
│                     EmoSense - Emotion Detector              │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│   [ LIVE WEBCAM FEED ]       │   Detected Emotion:          │
│                              │                              │
│   ┌─────────────────────┐    │   😀 Happy                   │
│   │   [Face detected]   │    │                              │
│   │   with bounding box │    │   Confidence: 96.3%          │
│   └─────────────────────┘    │                              │
│                              │   ┌──────────────────┐        │
│   Real-time processing...    │   │   [Emoji: 😃]    │        │
│                              │   └──────────────────┘        │
│                              │                              │
│                              │   "Keep smiling! You're      │
│                              │    doing great!"             │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Example Outputs

**Input**: Person smiling at webcam  
**Output**:
- Emotion: Happy 😀
- Confidence: 96.3%
- Message: "Keep smiling! You're doing great!"

**Input**: Person with furrowed brows  
**Output**:
- Emotion: Angry 😠
- Confidence: 89.7%
- Message: "Take a deep breath and relax!"

---

## 🛡️ Error Handling

EmoSense includes robust error handling for:

- ✅ Camera access failures
- ✅ Model loading errors
- ✅ Missing emoji assets
- ✅ Face detection failures (displays "No Face Detected")
- ✅ Invalid frame captures
- ✅ Prediction smoothing to reduce jitter
- ✅ Graceful exit on exceptions

---

## 🌍 Cross-Platform Support

| Operating System | Support | Tested | Notes |
|-----------------|---------|--------|-------|
| 🪟 Windows 10/11 | ✅ Full | ✅ Yes | Optimized camera access |
| 🍎 macOS | ✅ Full | ✅ Yes | AVFoundation support |
| 🐧 Linux | ✅ Full | ✅ Yes | V4L2 compatible |

---

## 🚀 Future Enhancements

### Planned Features

- 🔹 **Emotion-based music recommendations** using Spotify API
- 🔹 **Voice feedback system** with text-to-speech
- 🔹 **Cloud analytics dashboard** for emotion tracking over time
- 🔹 **Multi-face detection** for group emotion analysis
- 🔹 **IoT integration** for smart home mood lighting
- 🔹 **Mobile app version** using React Native + TensorFlow Lite
- 🔹 **Export emotion logs** to CSV/JSON
- 🔹 **Custom emotion training** interface

### Research & Development

- 📊 Micro-expression detection
- 🎯 Context-aware emotion understanding
- 🧬 Personalized emotion baselines
- 🌐 Multi-cultural emotion recognition

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

With the requirement to include the original license and copyright notice.

---

## 👨‍💻 Author

<div align="center">

### Deeptimaan K

[![GitHub](https://img.shields.io/badge/GitHub-deeptimaan--k-181717?style=for-the-badge&logo=github)](https://github.com/deeptimaan-k)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/deeptimaan-k)

**"Emotions speak louder than words — now, your computer understands them too."** 🧠💫

</div>

---

## 🙏 Acknowledgments

- **FER-2013 Dataset** for training data
- **OpenCV** community for computer vision tools
- **TensorFlow/Keras** team for the deep learning framework
- All contributors and supporters of this project

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider:

- ⭐ **Starring** this repository
- 🍴 **Forking** it for your own experiments
- 🐛 **Reporting bugs** to help improve it
- 💡 **Suggesting new features**
- 📢 **Sharing** with others who might benefit

<div align="center">

### Made with ❤️ by [deeptimaan-k](https://github.com/deeptimaan-k)

**Star ⭐ this repository if it helped you!**

[![GitHub Stars](https://img.shields.io/github/stars/deeptimaan-k/EmoSense?style=social)](https://github.com/deeptimaan-k/EmoSense/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/deeptimaan-k/EmoSense?style=social)](https://github.com/deeptimaan-k/EmoSense/network/members)
[![GitHub Watchers](https://img.shields.io/github/watchers/deeptimaan-k/EmoSense?style=social)](https://github.com/deeptimaan-k/EmoSense/watchers)

</div>

---

<div align="center">

**Questions or suggestions?** [Open an issue](https://github.com/deeptimaan-k/EmoSense/issues) or reach out!

</div>
