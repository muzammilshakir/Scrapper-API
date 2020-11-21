import keras
import os
import audioread
import soundfile as sf
import warnings
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


warnings.filterwarnings('ignore')  # ignore warning when mp3 used

# parsing


# Model
model = load_model('model/weights.hdf5')

# Labels
classes = ['B', 'A', 'U', 'S', 'F', 'D', 'H', 'C']
classes = ['balanced', 'angry', 'surprise', 'sad', 'fear', 'disgust', 'happy', 'contempt']
# classes = ['N', 'A', 'U', 'S', 'F', 'D', 'H', 'C']
# Primary emotions and the consensus emotion code:
# Neutral	(N) (RENAME TO BALANCED)
# Angry		(A)
# Surprise	(U)
# Sad		(S)
# Fear		(F)
# Happy		(H)
# Disgust	(D)
# Contempt	(C)

le = LabelEncoder()
le.fit(classes)


def __extract_feature(file_name):
    try:
        # audio_data, sample_rate = sf.read(os.fspath(file_name))
        audio_data, sample_rate = librosa.load(os.fspath(file_name), res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(e)
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([mfccsscaled])


def predict_sound_emotion(file_name):
    prediction_feature = __extract_feature(file_name)
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]

    results = {}
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        results[category[0]] = round(predicted_proba[i] * 100, 2)
    return results
