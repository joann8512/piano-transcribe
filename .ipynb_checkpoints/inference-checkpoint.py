from piano_transcription_inference.inference import PianoTranscription
from piano_transcription_inference import config
from piano_transcription_inference.utilities import load_audio
import os
import time
import glob
import torch
import argparse
import tqdm

def traverse_dir(path):
    all_path = []
    for root, dirs, files in os.walk(path):
        for name in files:
            folder = name.split('_')[0]
            all_path.append(os.path.join(root, name))
    return all_path

def main():
    print('CUDA Availablility: ', torch.cuda.is_available())
    print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))
    st = time.time()
    
    # Arugments & parameters
    audio_path = traverse_dir(args.audio_path)
    output_midi_path = args.output_midi_path
    if not os.path.exists(output_midi_path):
        os.makedirs(output_midi_path)
        
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    
    
    for i, song in enumerate(audio_path):
        print('Processing [{}/{}] {}'.format(i, len(audio_path), song))
        title = song.split('/')[-1].split('.')[0]#.replace(' ', '_')
        title += '.mid'
        if title in os.listdir(output_midi_path):  # Check if already processed
            print('Passing ', song)
            continue
            
        # Load audio
        #print("### Loading ###")
        #print(song)
        else:
            (audio, _) = load_audio(song, sr=config.sample_rate, mono=True)

            # Transcriptor
            #print("### Transcriptor ###")
            transcriptor = PianoTranscription(device=device)

            # Transcribe and write out to MIDI file
            #print("### Transcribing/Writing ###")
            transcribed_dict = transcriptor.transcribe(audio, os.path.join(output_midi_path, title))
    ed = time.time()

    print("### DONE ###")
    print("Took "+str(ed-st)+" seconds to process.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio_path', default='../test',type=str)
    parser.add_argument('--output_midi_path', default='../transcribed', type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)

    args = parser.parse_args()
    main()
    
    #audio_path = ../piano-emotion-src/compound-word-transformer/dataset_wayne/remy_17k_audio_data/data/mp3/DooPiano
    #output = ../piano-emotion-src/compound-word-transformer/dataset_wayne/remy_17k_audio_data/pedal_midi/src_002
    #check = '../piano-emotion-src/compound-word-transformer/dataset_wayne/remy_17k_audio_data/pedal_midi/src_002'
    