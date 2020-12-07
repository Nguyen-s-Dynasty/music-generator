import pickle
import os
import sys
from progress.bar import Bar
import utils
from midi_processor.processor import encode_midi
from mido.midifiles.meta import KeySignatureError
from mido import MidiFile
import numpy as np

def preprocess_midi(path):
    return encode_midi(path)


def preprocess_midi_files_under(midi_root, save_dir):
    total_secs = []
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    i = 0
    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
            total_secs.append(MidiFile(path).length)
        except KeyboardInterrupt:
            print('Abort')
            return
        except EOFError:
            print('EOF Error')
            continue
        except:
            print()
            print('Error in decoding: {}'.format(path))
            continue
            
        if len(data) > 0:
            i+= 1
            with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
                pickle.dump(data, f)
    print(f'Successfully Processed {i} files!') 
    print(f'Analyzed {len(total_secs)} files!')
    print('Dataset Descriptives in Minutes):')
    total_secs = np.array(total_secs)
    print(f'Sum: {total_secs.sum()}\tMean: {total_secs.mean()/60.0}\nMin: {total_secs.min()/60.0}\tMax: {total_secs.max()/60.0}\nStdev: {total_secs.std()/60.0}')
if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
