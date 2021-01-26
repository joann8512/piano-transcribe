from piano_transcription_inference.inference import PianoTranscription
from piano_transcription_inference import config
from piano_transcription_inference.utilities import load_audio
import os
import time
import glob
import torch
import argparse


def audio2MIDI(song, output, device):
    st = time.time()
    
    # Load audio
    print("### Loading ###")
    (audio, _) = load_audio(song, sr=config.sample_rate, mono=True)

    # Transcriptor
    print("### Transcriptor ###")
    transcriptor = PianoTranscription(device=device)

    # Transcribe and write out to MIDI file
    # Check if output directory exists
    print("### Transcribing/Writing ###")
    if not os.path.exists(output):
        os.makedirs(output)
    name = song.split(".")[1].split("/")[-1]  # name == filename to save
    transcribed_dict = transcriptor.transcribe(audio, os.path.join(output, name+".mid"))
    ed = time.time()

    print("### DONE ###")
    print("Took "+str(ed-st)+" seconds to process.")
    
    #return os.path.join(output, name+".mid")
    
    
if __name__ == "__main__":
    audio2MIDI()
