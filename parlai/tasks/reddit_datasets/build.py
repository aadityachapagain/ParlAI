from parlai.gcp.gcs_service import gcp
import os
import re
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

RESOURCES = 'reddit/20200827/'
file_name = 'train-00001-of-00005.txt'
matcher = '{}-0000[0-4]-of-00005.txt'

def build(opt):
    dpath = os.path.join(opt['datapath'], 'reddit_datasets')
    dtype = opt.get('datatype', 'train')
    regex = re.compile(matcher.format(dtype))
    # check if file already exists or not
    exists = True if os.path.isdir(dpath) else False
    try:
        condt = any([re.fullmatch(regex, f) != None for f in os.listdir(dpath)])
    except:
        condt = False
    if exists and condt:
        files_list = os.listdir(dpath)
        if any([re.fullmatch(regex, f) == None for f in files_list]):
            req_files = [ '{}-0000{}-of-00005.txt'.format(dtype,i) for i in range(5)]
            req_files_table = [(req_file, req_file in files_list) for req_file in req_files]
            for index_fl in req_files_table:
                if not index_fl[1]:
                    gcp.download(os.path.join(RESOURCES, index_fl[0]), dpath)
    else:
        gcp.download_all(RESOURCES, dpath)
    build_data.mark_done(dpath)