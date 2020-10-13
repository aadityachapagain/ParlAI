from parlai.gcp.gcs_service import gcp
import os
import re
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

def build(opt):
    dpath = os.path.join(opt['datapath'], 'unliked_utterances')
    dtype = 'train' if 'train' in opt.get('datatype','train') else 'valid'
    if not os.path.isfile(f'{dpath}/train.txt'):
        gcp.download('unliked_utterances/train.txt',dpath)
    if not os.path.isfile(f'{dpath}/valid.txt'):
        gcp.download('unliked_utterances/valid.txt',dpath)
    
    build_data.mark_done(dpath)

if __name__ == "__main__":
    build({'datapath': 'data', 'datatype': 'train:stream'})
