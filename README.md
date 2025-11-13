# ucf-101_visionpp

A lightweight **Python-based video classification system** that recognizes human actions (e.g., *Archery*, *BasketballDunk*, *PlayingPiano*) from the **UCF-101 dataset**.  
It extracts and normalizes frames, then classifies actions using either a **custom neural network** or a **distance-based method**, built entirely with NumPy and OpenCV.

---

## Overview

This project demonstrates video action recognition without relying on TensorFlow or PyTorch.  
It processes videos frame by frame, trains a small neural model, and predicts actions based on learned visual patterns.  
Developed quickly as an experiment, it serves as a practical introduction to computer vision and neural network fundamentals.

---

## Features

🎞️ Frame extraction and grayscale normalization  
🧠 Simple NumPy-based neural network  
📏 Distance-based baseline classifier  
📦 `.pkl` dataset serialization for reuse  
🔍 Educational, no heavy ML frameworks required  

---

## Installation

```bash
git clone https://github.com/Lundii1/ucf-101_visionpp.git
cd ucf-101_visionpp
pip install numpy opencv-python pillow scipy
```

---

## Usage

```python
from vision_machine import VideoClassifier

# Create dataset from videos
VideoClassifier.createDatabase("UCF-101/TestFolder", "exampleset")

# Train model
clf = VideoClassifier("exampleset")
clf.openDatabase()
clf.train(learning_rate=0.0001, repetitions=10)

# Classify a test video
result = clf.classifyVideoNN("test_set/v_Archery_g01_c01.avi", clf.normalizeBinary)
print(result)
```

---

## Requirements

- Python 3.10+  
- NumPy, OpenCV, Pillow, SciPy  

---

## Dataset

Uses a subset of the **[UCF-101 Human Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php)**  
Example classes: Archery, BasketballDunk, HorseRace, PlayingPiano, PlayingViolin, Surfing  

---

## Future Work

- Integrate TensorFlow or PyTorch  
- Add 3D CNNs or RNNs for temporal modeling  
- Implement accuracy and performance metrics  

---

## Acknowledgments

Thanks to the **UCF Center for Research in Computer Vision**, and the developers of **OpenCV** and **NumPy** for enabling accessible computer vision research.
