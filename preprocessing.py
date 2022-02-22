'''
quick script to take .m4a data directly from the vox dataset (in ./dataset/raw/) and convert to mono WAV
format. Can also include further preproessing steps here if necessary.
'''

from pydub import AudioSegment
import os
from tqdm import tqdm
os.system('conda install -c main ffmpeg') # need this for pydub to function


def mkdir_if_not_exists(path):
    '''

    Args:
        path: path to check the existance of. Must be a directory path without a file at end

    Returns: None, but creates the path if it doesn't exist

    '''
    path_splits = path.split('/')
    path_splits.remove('.')
    incremental_path = '.' # we need to iterate through all the subdirectories in 'path' to incrememtally create them
    for subpath in path_splits:

        incremental_path = os.path.join(incremental_path, subpath) # build the incrememtal path, check if it exists and build
        if not os.path.exists(incremental_path):
            os.mkdir(incremental_path)
        else:
            pass

    return


def m4a_to_wav(input_path, output_path):
    '''

    Args:
        input_path: path to the m4a target file
        output_path: path where the wav file will be exported

    Returns: None

    '''

    track = AudioSegment.from_file(input_path, format='m4a') # read the m4a file
    file_handle = track.export(output_path, format='wav') # export the wav
    return


def main():
    '''

    Scans through all subdirectories of ./dataset/raw/, and recreates them in ./dataset/processed/ (writing new
    directories if needed), and converting the m4a files into wav format
    '''

    rootdir = './dataset/raw/'
    outdir = './dataset/processed/'
    ids = os.listdir(rootdir)
    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids): # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(rootdir, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(rootdir, id, ctx)
            procpath = os.path.join(outdir, id, ctx)
            mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                    m4a_to_wav(os.path.join(rawpath, f),
                               os.path.join(procpath, f[:-3] + 'wav'))
    return


if __name__ == '__main__':
    main()
