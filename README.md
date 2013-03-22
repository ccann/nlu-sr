# KNN Decoder for Speech Recognition

@author: ccann

This is an incredibly simple speech recognizer built for CS 150-06: Situated Natural
Language Understanding on Robots at Tufts University during my M.S. degree. It reads
speech from a file (.wav) and outputs the recognized utterance. Right now the dictionary
of words includes three color words: red, green and white.

## Decoding

- Divide the utterance (e.g. "red") into N bins. Each bin represents a portion of the
  utterance. "red" ==> [ bin1 -- bin2 -- bin3 -- ... -- binN ]

- Each bin contains "meta features" derived from acoustic features themselves derived from
  a portion of the input utterance. The acoustic features are supplied by the Sphinx4
  frontend, specifically the DeltasFeatureExtractor which supplies the delta and double
  delta of the cepstra. The meta features in this decoder are computed from *just* the
  cepstra. Currently the delta and double delta are ignored.

- Meta feature implementation is currently VERY naive. Each meta-feature vector consists
  of the following: mean, min, max, stdev, and variance of the cepstra.

- A k-nearest-neighbors (KNN) classifier (using WEKA's IBk) is trained on each individual
  bin, i.e. on the features of each portion (1/5 if NUM_BINS == 5) of each utterance in
  the training set.

- During testing each classifier classifies a portion of the test utterance as one of the
  words in the dictionary (e.g. red, green, or white). The cumulative classification tally
  for each dictionary word is retained in the scores list. The KNNDecoder determines the
  most likely classification of the whole utterance based on the dictionary word with the
  highest score (i.e. highest number of respective bin classifications) relative to the
  others.

