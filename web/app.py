from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow.keras as keras
import librosa
import math
import numpy as np

app = Flask(__name__)
our_model = keras.models.load_model('models/cnn_model_77acc_lr15.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=["POST"])
def predict():
    data = request.files[0]
    input_file = create_mfcc(data)
    prediction = our_model.predict(our_model, input_file)
    return jsonify(predictions=prediction)


def create_mfcc(data, num_segments=5, hop_length=512, n_fft=2048, sample_rate=22050, num_mfcc=13):
    # arrays to hold calculated mfccs and mapped number
    mfccs = []
    labels = []

    samples_per_track = sample_rate * 30  # track duration = 30s
    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # mapping of genre to number
    genre_dict = {
        'jazz': 0,
        'reggae': 1,
        'rock': 2,
        'blues': 3,
        'hiphop': 4,
        'country': 5,
        'metal': 6,
        'classical': 7,
        'disco': 8,
        'pop': 9
    }

    signal, sr = librosa.core.load(data, sr=sample_rate)

    # use genre filename to map number to MFCC for training
    label = str(data).split('.')[0]
    label_mapped = genre_dict[label]

    # extract MFCCs
    for s in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * s
        finish = start + samples_per_segment
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        # transpose to correct dimensions
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfccs.append(mfcc.tolist())
            labels.append(label_mapped)

    # convert to numpy arrays for processing
    np_mfccs = np.array(mfccs)
    np_genres = np.array(labels)

    return np_mfccs, np_genres


def predict(model, X):
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]  # array shape (1 <- number of samples, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    # predicted_index = np.argmax(prediction, axis=1)

    return prediction


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)