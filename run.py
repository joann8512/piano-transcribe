import os
import shutil
import argparse
import glob
import torch
import random
import pandas as pd

from transcribe import audio2MIDI

# Transcribe mp3 to MIDI
def transcribe(song_list, output, device):  # song_list, output, device
    for audio_path in song_list:
        midi = audio2MIDI(audio_path, output_midi_path, device)  # Transcribe, return midi path

def main():
    # Arugments & parameters
    audio_path = glob.glob(args.audio_path)
    output_midi_path = args.output_midi_path
    midi_clean = args.output_midi_clean
    device = 'cuda:4' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    transcribe(audio_path, output_midi_path, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio_path', type=str, required=False, default="./audios/seg/*.mp3")
    parser.add_argument('--output_midi_path', type=str, required=False, default="./audios/midi/")
    parser.add_argument('--cuda', action='store_true', required=True)

    args = parser.parse_args()
    main()
