from parlai.gcp.gcs_service import gcp
import os
import re
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

gcp_data_path = 'reddit_processed_feb_15/omit_maybe'
matcher = 'clean-{}-00[01][0-9][0-9]-of-00200.txt'
data_lenghts_count = 'reddit_processed_feb_15/omit_maybe/train_data.lengths'

def build(opt):
    RESOURCES = opt.get('gcs_data_path', gcp_data_path)
    dpath = os.path.join(opt['datapath'], 'reddit_cleaned_omit_maybe_datasets/train_data')
    dtype = 'train' if 'train' in opt.get('datatype','train') else 'valid'
    
    regex = re.compile(matcher.format(dtype))
    # check if file already exists or not
    exists = os.path.isdir(dpath)
    try:
        condt = any([re.fullmatch(regex, f) != None for f in os.listdir(dpath)])
    except:
        condt = False
    if exists and condt:
        files_list = os.listdir(dpath)
        if any([re.fullmatch(regex, f) == None for f in files_list]):
            req_files = ['clean-{}-00{:03}-of-00200.txt'.format(dtype,i) for i in range(200)]
            req_files_table = [(req_file, req_file in files_list) for req_file in req_files]
            for index_fl in req_files_table:
                if not index_fl[1]:
                    gcp.download(os.path.join(RESOURCES, index_fl[0]), dpath)
    else:
        gcp.download_all(RESOURCES, dpath)
    if not os.path.isfile(os.path.join(opt['datapath'], 'reddit_cleaned_omit_maybe_datasets', 'train_data.lengths')):
        gcp.download(data_lenghts_count,os.path.join(opt['datapath'], 'reddit_cleaned_omit_maybe_datasets', ))
    
    build_data.mark_done(dpath)

if __name__ == "__main__":
    build({'datapath': 'data', 'datatype': 'train:stream'})
