from flask import Flask,request
import requests
from cdifflib import CSequenceMatcher
import hashlib
import os
import json
import time
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import librosa
from tensorflow.keras.models import load_model


def combine_features(data):
  features = []
  for i in range(0, data.shape[0]):
    features.append(data['name'][i]+" "+ data['genre'][i]+" "+ data['singer'][i] )

  return features

def read_file(url, chunk_size=5242880):
    response = requests.get(url, stream=True)
    for chunk in response.iter_content(chunk_size=chunk_size):
        yield chunk

def upload_file(api_token, url):

    print(f"Uploading file from URL: {url}")

    headers = {'authorization': api_token}


    response = requests.post('https://api.assemblyai.com/v2/upload',
                             headers=headers,
                             data=read_file(url))

    if response.status_code == 200:
        return response.json()["upload_url"]

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def create_transcript(api_token, audio_url):

    print("Transcribing audio... This might take a moment.")

    url = "https://api.assemblyai.com/v2/transcript"

    headers = {
        "authorization": api_token,
        "content-type": "application/json"
    }


    data = {
        "audio_url": audio_url
    }


    response = requests.post(url, json=data, headers=headers)

    transcript_id = response.json()['id']

    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            break

        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")


        else:
            time.sleep(3)

    return transcription_result
# Your API token is already set in this variable
your_api_token = "b782cb0c9f8f4620af5ba8a6d7ff0702"


SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)




app = Flask(__name__)

@app.route("/api/recommend/" , methods=['GET'])
def final_function():
    lst = []
    name = []
    try:
        bar = request.args.to_dict()
        print(bar)
        link = bar.get("link")
        title = bar.get("title")
        out_file1 = "song_data.csv"
        resp = requests.get(link)
        resp.raise_for_status()
        with open(out_file1, "wb") as fout:
            fout.write(resp.content)

        df = pd.read_csv("song_data.csv", encoding="unicode_escape")

        coloums = ["name", "genre", "singer", "played"]

        df['combined_features'] = combine_features(df)

        cm = CountVectorizer().fit_transform(df['combined_features'])

        cs = cosine_similarity(cm)


        song_id = df[df.name == title]['id'].values[0]

        scores = list(enumerate(cs[song_id]))

        sorted_scores = sorted(scores, key = lambda x:x[1], reverse= True)
        sorted_scores = sorted_scores[1:]

        i=0

        for item in sorted_scores:
            lst.append(item[0])
            #song = df[df.id == item[0]]['name'].values[0]
            #print(i+1, song)
            i+=1
            if i>=10:
                break
        for i in lst:
            n = df.loc[df['id'] == i, 'name'].values[0]
            name.append(n)
    except:
        for i in range(10):
            lst.append(random.randint(1,49))
        for i in lst:
            n = df.loc[df['id'] == i, 'name'].values[0]
            name.append(n)
    print(name)


    os.remove("song_data.csv")
    return name


@app.route('/api/plag/', methods=['GET'])
def foo(out_file1="audio1.mp3", out_file2="audio2.mp3"):
  bar = request.args.to_dict()
  print(bar)
  url1 = bar.get("a")
  url2 = bar.get("b")
  print(url1)
  print(url2)
  out_file1 = "audio1.mp3"
  resp = requests.get(url1)
  resp.raise_for_status()
  with open(out_file1, "wb") as fout:
      fout.write(resp.content)

  out_file2 = "audio2.mp3"
  resp = requests.get(url2)
  resp.raise_for_status()
  with open(out_file2, "wb") as fout:
      fout.write(resp.content)

  h1 = hashlib.sha1()
  h2 = hashlib.sha1()

  with open("audio1.mp3", "rb") as file:
    chunk = 0
    while chunk != b'':
      chunk = file.read(1024)
      h1.update(chunk)

  with open("audio2.mp3", "rb") as file:
    chunk = 0
    while chunk != b'':
      chunk = file.read(1024)
      h2.update(chunk)

  msg1 = h1.hexdigest()
  msg2 = h2.hexdigest()
  os.remove("audio1.mp3")
  os.remove("audio2.mp3")

  r = (CSequenceMatcher(None,msg1,msg2).ratio())*100
  r=int(r)
  r = [r]
  print(r)
  return r


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribe audio from a URL using AssemblyAI API.

    Expects a JSON payload in the request body with the following format:
    {
        "api_token": "YOUR_API_TOKEN",
        "audio_url": "URL_TO_AUDIO_FILE"
    }

    Returns:
        dict: Completed transcript object.
    """
    data = request.get_json()
    api_token = data["api_token"]
    audio_url = data["audio_url"]

    # Upload the file to AssemblyAI and get the upload URL
    upload_url = upload_file(api_token, audio_url)

    if upload_url:
        # Transcribe the audio file using the upload URL
        transcript = create_transcript(api_token, upload_url)
        return json.dumps(transcript)
    else:
        return "Error uploading file."



@app.route("/api/genre/<path:url>")
def inst(url):
    out_file = "store_3mp/eee/audio.mp3"
    resp = requests.get(url)
    resp.raise_for_status()
    with open(out_file, "wb") as fout:
        fout.write(resp.content)

    DATASET_PATH = "store_3mp"
    JSON_PATH = "starboy.json"
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    modd = load_model('bitch.h5', compile=False)
    with open('starboy.json', "r") as fp:
      data = json.load(fp)

    X = np.array(data["mfcc"])

    prediction = modd.predict(X)

    print(prediction)

    predicted_index = np.argmax(prediction, axis=1)

    print("******************************************************************")
    print(predicted_index[0])

    os.remove("store_3mp/eee/audio.mp3")
    os.remove("starboy.json")

    t = str(predicted_index[0])

    return t



if __name__ == '__main__':
  app.run(host="0.0.0.0",port=8080,debug=True)

