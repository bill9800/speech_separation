# speech_separation

This is a repository for speech separation tasks. 

This project is highly inspired by the paper[1], and is still working to improve the performance.

# Data

[AVspeech dataset](https://looking-to-listen.github.io/) : contains 4700 hours of video segments, from a total of 290k YouTube videos.

Customized video and audio downloader provided in data. (based on youtube-dl,sox,ffmpeg)  

# Preprocessing

There are several preprocess functions in the lib. Including STFT, iSTFT, power-law compression, complex mask etc.

Apply MTCNN to detect face and correct it by checking the provided face center. [2]

The visual frames are transfered to 1792 (avg pooling layer) face embeddings with facenet pre-trained model[3].

# Model

Audio part : Dilated CNN + Bidirectional LSTM.

Video part : (pretrained MTCNN + Facenet) + dilated CNN + Bidirectional LSTM.

Loss function : modified discriminative loss function inspired from paper[4].

# Prediction

Apply complex ratio mask (cRM) to enhance phase spectrum. Maintain the quality during transformation by hyperbolic tangent fucntion.[4]

The model will be evaluated by signal-to-distortion ratio.


# Reference

[1] [Lookng to Listen at the Cocktail Party:A Speaker-Independent Audio-Visual Model for Speech Separation, A. Ephrat et al., arXiv:1804.03619v2 [cs.SD] 9 Aug 2018](https://arxiv.org/abs/1804.03619)

[2] [MTCNN face detection](https://github.com/ipazc/mtcnn)

[3] [FaceNet Pretrained model](https://github.com/davidsandberg/facenet)

[4] [Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation, P. Hunag et al,arXiv:1502.04149v4 [cs.SD] 1 Oct 2015](https://arxiv.org/abs/1502.04149)

[5] [Complex Ratio Masking for Monaural Speech Separation](https://ieeexplore.ieee.org/document/7364200)
