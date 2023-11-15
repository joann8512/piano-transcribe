from piano_transcription_inference.inference import PianoTranscription
from piano_transcription_inference import config
from piano_transcription_inference.utilities import load_audio
import os
import time
import torch
import argparse

def traverse_dir(
        root_dir,
        extension=('wav'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=True,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list

def main():
    print('CUDA Availablility: ', torch.cuda.is_available())
    print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))
    st = time.time()
    
    # Arguments & parameters
    audio_path = traverse_dir(args.audio_path)
    output_midi_path = args.output_midi_path
    if not os.path.exists(output_midi_path):
        os.makedirs(output_midi_path)
        
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    
    
    for i, song in enumerate(audio_path):
        print('Processing [{}/{}] {}'.format(i, len(audio_path), song))
        title = os.path.splitext(song.split('/')[-1])[0]
        title += '.midi'
        if title in os.listdir(output_midi_path):  # Check if already processed
            print('Passing ', song)
            continue
            
        # Load audio
        else:
            (audio, _) = load_audio(song, sr=config.sample_rate, mono=True)

            # Transcriptor
            transcriptor = PianoTranscription(device=device)

            # Transcribe and write out to MIDI file
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
    