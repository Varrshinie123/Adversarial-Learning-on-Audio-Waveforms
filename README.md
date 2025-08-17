# 🎵 Adversarial Learning on Audio Waveforms

This project explores the robustness of **deep learning models for audio classification** under adversarial perturbations.  
We train a **CNN-based classifier** on the [SpeechCommands dataset](https://arxiv.org/abs/1804.03209) and evaluate it against **FGSM adversarial attacks** on raw audio waveforms.

---

## 🔑 Features
- **Audio Preprocessing**
  - Log-Mel spectrogram extraction  
  - Automatic padding/cropping and resampling  
- **CNN Classifier**
  - 3 convolutional layers with batch normalization and pooling  
  - Dense layers for classification on keywords (*yes, no, up, down*)  
- **Adversarial Attacks**
  - Implements **FGSM (Fast Gradient Sign Method)** at the waveform level  
  - Evaluates clean vs. adversarial accuracy across multiple epsilon values  
- **Visualization & Audio Playback**
  - Listen to clean vs. adversarial samples  
  - Plot waveforms and spectrograms to inspect perturbations  

---

## 📊 Results
- Achieves strong accuracy on clean SpeechCommands samples  
- FGSM perturbations cause **sharp drops in accuracy**, even when noise is barely perceptible  
- Highlights the **vulnerability of speech models** to adversarial examples  

---

## 🚀 Getting Started

### Install dependencies
```bash
pip install torch torchaudio matplotlib TTS moviepy

### Train Classifier
python train.py

### Run adversarial attack
python attack_fgsm.py


