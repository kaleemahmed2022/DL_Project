import preprocessing
import pandas as pd

i = 4

files = pd.read_csv('Index_{}.csv'.format(i))['0'].to_list()[681:]

preprocessing.dataset_to_wav('/Users/jameswilkinson/Downloads/dev/aac/',
                            '/Users/jameswilkinson/Downloads/dev/wav/',
                             filenames = files)
