from parlai.gcp.gcs_service import gcp
import os
import re
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

def build(opt):
    dpath = os.path.join(opt['datapath'], 'moxie_grl')
    dtype = 'train' if 'train' in opt.get('datatype','train') else 'valid'
    if not os.path.isfile(f'{dpath}/train.txt'):
        gcp.download((f'moxie_data/train.txt',dpath)
    if not os.path.isfile(f'{dpath}/valid.txt'):
        gcp.download((f'moxie_data/valid.txt',dpath)
    
    build_data.mark_done(dpath)

if __name__ == "__main__":
    build({'datapath': 'data', 'datatype': 'train:stream'})
