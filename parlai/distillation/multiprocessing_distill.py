#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Main launch script for single-host, multi-GPU training.

This is a drop-in replacement for train_model.py.  This script will launch N
subprocess, each which runs the full training loop independently.

Uses torch.nn.parallel.DistributedDataParallel for its main uses.  Agents must
specifically implement the wrapper of DistributedDatParallel, but all
TorchRankerAgents and TorchGeneratorAgents support this.

Examples
--------

.. code-block:: shell

  parlai multiprocessing_train -m transformer/generator -bs 16 -t convai2 -mf /tmp/mymodel
"""

import faulthandler; faulthandler.enable()

import torch
import yaml
import random
import os
import signal
import parlai.distillation.distill_model as single_train
from parlai.gcp.gcs_service import gcp
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script

def get_latest_train(file_path):
    try:
        cand = list(set([ os.path.join(*os.path.split(i)[:1]) for i in gcp.list_files(file_path) if os.path.split(i)[1].strip() !='']))
        cand = [i for i in cand if '.tensorboard' not in i ]
        cand = {int(i.split('_')[-1]):i for i in cand}
        latest = sorted(list(cand.keys()), reverse=True)[0]
        latest = cand[latest]
        return latest
    except:
        return False

def multiprocess_train(
    rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'
):
    with distributed_utils.distributed_context(
        rank, opt, port, rank_offset, gpu, hostname
    ) as opt:
        # Run the actual training
        return single_train.TrainLoop(opt).train()


def launch_and_train(opt, port):
    """
    Perform a fork() to many processes.
    """
    # First get model from gcs
    latest_train_path = get_latest_train(opt['run_tag'])

    model_download_path = os.path.join(*os.path.split(opt['student_model_file'])[:-1])
    if latest_train_path:
        if not os.path.isfile(opt['student_model_file']+'.checkpoint'):
            gcp.download_all(latest_train_path, model_download_path)
    
    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.spawn(
        multiprocess_train,
        # need to give rank offset as 1 to cover the fact that the main
        # process is rank 0, but that spawn() doesn't let you control rank
        (opt, port, 1),
        nprocs=opt['distributed_world_size'] - 1,  # main proc will also run loop
        join=False,
    )

    try:
        retval = multiprocess_train(0, opt, port)
        spawncontext.join()
        return retval
    except KeyboardInterrupt:
        # tell the subprocesses to stop too
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)
        raise


def setup_args():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    return parser


@register_script("multiprocessing_train", aliases=["mp_train"], hidden=True)
class MultiProcessTrain(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        if self.opt.get('config_path', False):
            print('config path Exists !')
            with open(self.opt['config_path']) as fp:
                configs = yaml.load(fp.read(), Loader=yaml.FullLoader)
                self.opt.update(configs)
                self.opt['student_config'] = configs['student_config']
                self.opt['teacher_config'] = configs['teacher_config']
        else:
            print('specify config_path ...')
        port = random.randint(32000, 48000)
        return launch_and_train(self.opt, port)


if __name__ == '__main__':
    MultiProcessTrain.main()
