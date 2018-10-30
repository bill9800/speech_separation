# speech_separation

This is a repository for speech separation tasks. 

This project is highly inspired by the paper[1], and is still working to improve the performance.

# Data

[AVspeech dataset](https://looking-to-listen.github.io/) : contains 4700 hours of video segments, from a total of 290k YouTube videos.

# Preprocessing

There are several preprocess functions in the lib. Including STFT, iSTFT, power-law compression etc.

The visual frames are transfered to 512 face embeddings with facenet pre-trained model[2].

# Model

Audio part : Dilated CNN + Bidirectional LSTM

Video part : Still working 

# Prediction

Apply complex ratio mask (cRM) to enhance phase spectrum.

The model will be evaluated by signal-to-distortion ratio.

# Reference

[1] [Lookng to Listen at the Cocktail Party:A Speaker-Independent Audio-Visual Model for Speech Separation, A. Ephrat et al., arXiv:1804.03619v2 [cs.SD] 9 Aug 2018](https://arxiv.org/abs/1804.03619)

[2] [FaceNet Pretrained model](https://github.com/davidsandberg/facenet)
