import librosa
import librosa.display

def get_mfcc_features(audio_file, fs=16000, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=fs) # the audio file is sampled at 16 kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # 13 is the typical number of coefficients
    return mfccs

