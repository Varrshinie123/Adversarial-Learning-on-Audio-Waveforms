# -Adversarial-Learning-on-Audio-Waveforms
This project explores the robustness of deep learning models for audio classification under adversarial perturbations. We train a CNN-based classifier on the SpeechCommands dataset and evaluate it against FGSM adversarial attacks on raw audio waveforms.

Features

Audio Preprocessing

Uses log-Mel spectrograms as CNN input

Supports automatic padding/cropping and resampling

CNN Classifier

Convolutional neural network with 3 conv layers, batch normalization, pooling, and dense layers

Trained to classify a subset of keywords (yes, no, up, down)

Adversarial Attacks

Implements FGSM (Fast Gradient Sign Method) at the waveform level

Evaluates clean vs. adversarial accuracy under varying epsilon perturbations

Visualization & Audio Playback

Listen to clean vs. adversarial samples

Plot waveforms and spectrograms to inspect adversarial noise

📊 Results

Achieves high accuracy on clean SpeechCommands samples

FGSM adversarial noise leads to a sharp drop in classifier accuracy, even with imperceptible perturbations

Demonstrates the vulnerability of audio models to adversarial attacks

🚀 How to Run (Colab)

Clone the repo or open the notebook in Google Colab.

Install dependencies:

pip install torch torchaudio matplotlib TTS moviepy


Run the training script:

python train.py


Run adversarial attack & evaluation:

python attack_fgsm.py

📌 Future Work

Extend to PGD and CW attacks for stronger adversarial robustness evaluation

Experiment with adversarial training to improve model resilience

Explore transferability of attacks across different architectures (RNNs, Transformers)

📖 Reference

Goodfellow et al., Explaining and Harnessing Adversarial Examples (2015)

Alzantot et al., Did you hear that? Adversarial Examples Against Automatic Speech Recognition (2018)
