from parlai.gcp.gcs_service import gcp
import os
import re
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

gcp_data_path = 'reddit/20200827/'
file_name = 'train-00001-of-00005.txt'
matcher = '{}-0000[0-4]-of-00005.txt'
data_lenghts_count = 'reddit/train_data.lengths'
data_valid_lenghts_count = 'reddit/train_data.lengths_valid'

def get_latest_train(file_path):
    try:
        cand = list(set([ os.path.join(*os.path.split(i)[:1]) for i in gcp.list_files(file_path) if os.path.split(i)[1].strip() !='']))
        cand = {int(i.split('_')[-1]):i for i in cand}
        latest = sorted(list(cand.keys()), reverse=True)[0]
        latest = cand[latest]
        return latest
    except:
        return False

def build(opt):
    RESOURCES = opt.get('gcs_data_path', gcp_data_path)
    dpath = os.path.join(opt['datapath'], 'reddit_datasets/train_data')
    dtype = 'train' if 'train' in opt.get('datatype','train') else 'valid'

    # First get model from gcs
    latest_train_path = get_latest_train(opt['run_tag'])

    model_download_path = os.path.join(*os.path.split(opt['student_model_file'])[:-1])
    if latest_train_path and not os.path.isfile(os.path.join(model_download_path, '.built')):
        gcp.download_all(latest_train_path, model_download_path)
        build_data.mark_done(model_download_path)
    
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
    if not os.path.isfile(os.path.join(opt['datapath'], 'reddit_datasets', 'train_data.lengths')):
        gcp.download(data_lenghts_count,os.path.join(opt['datapath'], 'reddit_datasets', ))

    if not os.path.isfile(os.path.join(opt['datapath'], 'reddit_datasets', 'train_data.lengths')):
        gcp.download(data_valid_lenghts_count,os.path.join(opt['datapath'], 'reddit_datasets', ))
    
    build_data.mark_done(dpath)