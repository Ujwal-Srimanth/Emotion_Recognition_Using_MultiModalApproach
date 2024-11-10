## Multimodal Emotion Recognition Using Audio and Text
#Overview
This project focuses on multimodal emotion recognition by leveraging both audio and text data. We implemented two different approaches to enhance the robustness and accuracy of emotion recognition models, exploring various architectures and techniques.

# Problem Statement
Emotion recognition is a critical task for many real-world applications like human-computer interaction, sentiment analysis, and customer service. Traditional models often rely on a single modality (either text or audio), which may not capture the full spectrum of emotional cues. By combining audio and text modalities, this project aims to create a more comprehensive emotion recognition model.

# Dataset
The dataset used for this project consists of multimodal samples with:

# Text data containing transcriptions of speech.
Audio data in the form of raw audio files TESS dataset.
Text datset is emotion dataset form dataset Library
Used SpeechRecognizer Library from google to extrcat the text from audio which helps while combining the approach
The text data was preprocessed using BertTokenizer, while audio features were extracted using librosa library. Features like MFCC, Chroma Means were extracted.

# Approach 1: Augmented Multimodal Fusion
# 1.1 Text Modality
Data Augmentation: Translated the text data into 5 different languages and translated it back to English. This provided variations and synonyms, making the text data more robust.
Model: Used TinyBERT, a lightweight and efficient model, for embedding text features.
# 1.2 Audio Modality
Data Augmentation: Applied various audio augmentations such as:
Time Stretching
Pitch Shifting
Noise Injection
Speed Change
Model: Used a 1D CNN model to extract features from the augmented audio data.
# 1.3 Fusion Strategy
Feature-Level Fusion: Combined the extracted features from TinyBERT (text) and 1D CNN (audio) using a Multi-Layer Perceptron (MLP).
Results: This approach showed moderate performance but had issues with overfitting due to extensive augmentations.
# Approach 2: Simplified Multimodal Fusion
# 2.1 Text Modality
Model: Used DistilBERT, a distilled version of BERT, without any data augmentations for text embeddings.
# 2.2 Audio Modality
Model: Used an LSTM model for processing the raw audio data without any augmentations. LSTM can capture temporal dependencies in the audio features effectively.
# 2.3 Fusion Strategy
Probability-Level Fusion: Combined the predicted probabilities from both models using argmax to select the final emotion label.
Results: This approach outperformed the first approach in terms of accuracy and generalization, demonstrating better stability and robustness.
