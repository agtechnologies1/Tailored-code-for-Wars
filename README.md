# What is this repository about?
Python script to automate vehicle logging coming and going from a video source or a live stream.

# To install dependencies
python -m venv env
Activate env ./myenv/Scripts/activate
add .env

VIDEO_SOURCE=
Has to be mp4, rts or an http(s) stream link

run the code
python vehicle_counter.py

it will log vehicles entering and leaving with timestamps in both csv and json format.

Take a look at vehicle_counter.csv or current_totals.json
