

import hashlib
from cdifflib import CSequenceMatcher
import requests
from pathlib import Path
import os
from flask import Flask,request



app = Flask(__name__)


@app.route("/")
def shit():
    return f"hello"


@app.route("/bullshit")
def shit():
    return f"yes u are shit"


@app.route('/api/foo/', methods=['GET'])
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


