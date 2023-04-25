

import hashlib
import json
import pickle
import numpy as np
import speech_recognition as sr
import time

from cdifflib import CSequenceMatcher
import requests
import os
from flask import Flask,request
from pydub import AudioSegment
from pydub.utils import make_chunks

app = Flask(__name__)


@app.route("/")
def shit():
    return f"hello"


@app.route("/bullshit")
def shit21():
    return f"yes u are shit"


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


@app.route("/api/audiotolyrics/", methods=['GET'])
def lyrics(out_file="audio.mp3"):
  os.system("sudo apt install ffmpeg")
  time.sleep(20)
  bar = request.args.to_dict()
  print(bar)
  url3 = bar.get("a")

  big = []
  text = ""
  #out_file = Path(f"/content/{out_file}").expanduser()

  resp = requests.get(url3)
  resp.raise_for_status()
  with open("audio.mp3", "wb") as fout:
    fout.write(resp.content)
  sound = AudioSegment.from_mp3("audio.mp3")
  sound.export("blindingLights.wav", format="wav")
  filename = "blindingLights.wav"
  # os.mkdir("/content")
  os.mkdir("temp")
  myaudio = AudioSegment.from_wav(filename)
  chunk_lenght_ms = 5000
  chunks = make_chunks(myaudio, chunk_lenght_ms)
  for i, chunk in enumerate(chunks):
    chunkName = 'temp/' + filename + "_{0}.wav".format(i)
    chunk.export(chunkName, format="wav")

  for files in sorted(os.listdir("temp")):
    filesss = "temp/" + files
    print(files)
    r = sr.Recognizer()

    # open the file
    with sr.AudioFile(filesss) as source:
      # listen for the data (load audio to memory)
      audio_data = r.record(source)
      # recognize (convert from speech to text)
      try:
        text = r.recognize_google(audio_data)
        print(text)
      except:
        print("...")
        text = "..."
    # small = [text]
    big.append(text)
  os.remove("audio.mp3")
  os.remove("blindingLights.wav")
  for files in sorted(os.listdir("temp")):
    filesss = "temp/" + files
    os.remove(filesss)
  os.rmdir("temp")
  # j = {"lyrics" : big}
  print(big)
  return big


@app.route("/genre")
def find_genre():
  loaded_mod = pickle.load(open('song_prediction.pickle', 'rb'))
  with open('testing.json', "r") as fp:
    data = json.load(fp)

  X = np.array(data["mfcc"])

  prediction = loaded_mod.predict(X)

  print(prediction)

  predicted_index = np.argmax(prediction, axis=1)

  print("******************************************************************")
  print(predicted_index[0])

  return predicted_index[0]



# @app.route("/api/audiotolyrics/", methods=['GET'])
# def lyrics(out_file="audio.mp3"):
#   bar = request.args.to_dict()
#   print(bar)
#   url3 = bar.get("a")
#
#   big = []
#   text = ""
#
#   resp = requests.get(url3)
#   resp.raise_for_status()
#   with open("audio.wav", "wb") as fout:
#     fout.write(resp.content)
#
#   r = sr.Recognizer()
#
#   # open the file
#   with sr.AudioFile("audio.wav") as source:
#     # listen for the data (load audio to memory)
#     audio_data = r.record(source)
#     # recognize (convert from speech to text)
#     try:
#       text = r.recognize_google(audio_data)
#       print(text)
#     except:
#       print("...")
#       text = "..."
#   x = text.split(" ",24)
#   os.remove("audio.wav")
#   return x
#


@app.route("/api/inst/")
def inst():
  os.system("sudo snap install ffmpeg")
  os.system('ffmpeg -version')






if __name__ == '__main__':
  app.run(host="0.0.0.0",port=5000,debug=True)

