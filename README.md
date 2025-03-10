# Gender and Speaker Classification

## Task 1: Gender Classification from audio files
1. Please find and download part of the LibriSpeech <a href="https://www.openslr.org/12/" target="_blank">here</a>. Find some time to read the
documentation of the database. Download the dev-clean corpus, which is relatively small in size. This database contains directories with ids. Each id represents one speaker and contains audio files associated with that speaker. You will also find a file that indicates the gender of each speaker.

2. In case you have computational issues, please feel free to select a subset of the dataset. However, try to keep at least 18 speakers (9 male and 9 female). The next step would be to extract features for the audio files. The most common type of features are the MFCC features. Feel free to use existing libraries in order to extract MFCC features for all the audio files of your dataset.

3. Once you have the MFCC features extracted, perform some data analysis to better
understand the data you are working with (features distribution, outliers detection, normalization techniques) and try to correlate them to the gender values.

4. After the previous step, you can start playing with various Machine Learning Algorithms. We would like you to perform a gender classification task. For each audio file you create a gender label. Build train and test (unseen while training) sets. Be careful when you split your dataset in train and test. Train a classifier on the train set and predict the labels of the test set. Measure accuracy and report performance. Try to experiment with at least 4 classifiers. Use two of them as baselines (one naive and one more sophisticated), a neural network and a deep learning model, such as a CNN. Once again, feel free to use any libraries that are convenient for you.

## Task 2: Speaker Classification from audio files
1. Now that you have completed the first task, let’s try something new. Read the
wav2vec2.0 <a href="https://maelfabien.github.io/machinelearning/wav2vec/#b-the-model" target="_blank">explanation</a> (if you are brave you can find and read the <a href="https://arxiv.org/pdf/2006.11477" target="_blank">paper</a>) and try to understand its basic and high-level concepts.

2. Download the wav2vec2.0 base model (No Finetuning) from this <a href="https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md" target="_blank">repo</a>, use it as <a href="https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#wav2vec" target="_blank">features extractor</a> to create features from audio signals. Apply the wav2vec model to the audio files you used in the previous tasks. We suggest you apply the model not on the entire file audio, but on 2 seconds subsequent chunks (though increasing the amount of samples).

3. We ask you to do two types of analysis over a speaker classification task:
    a. A quantitative analysis, using prediction models (starting from the features you just extracted). As before, use objective metrics (such as precision, recall and accuracy) to evaluate your results. Be careful when you split your dataset in train and test, it can be tricky.

    b. A qualitative analysis, using algorithms of dimensionality reduction and clustering and visualizing the results.
