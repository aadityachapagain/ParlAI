from parlai.gcp.gcs_service import gcp as storage_agent
import argparse
import traceback
import os


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-tag', required=True, help= 'get model checkpoint  of specific run tag')
    parser.add_argument('--model-file', required= True, help= 'name and path of the checkpoint to save ')
    return parser

def get_latest_train(file_path):
    try:
        cand = list(set([ os.path.join(*os.path.split(i)[:1]) for i in storage_agent.list_files(file_path) if os.path.split(i)[1].strip() !='']))
        cand = {int(i.split('_')[-1]):i for i in cand}
        latest = sorted(list(cand.keys()), reverse=True)[0]
        latest = cand[latest]
        return latest
    except:
        traceback.print_exc()
        return False


if __name__ == "__main__":
    opt = setup_parser().parse_args()
    latest_train_path = get_latest_train(opt.run_tag)
    if latest_train_path:
        if not os.path.isfile(opt.model_file +'.checkpoint'):
            storage_agent.download_all(latest_train_path, os.path.join(*os.path.split(opt.model_file)[:-1]))