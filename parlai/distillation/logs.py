#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Log metrics to tensorboard.

This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom.

.. code-block: none

   tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import os
import json
import numbers
from parlai.core.opt import Opt
from parlai.core.metrics import Metric
import parlai.utils.logging as logging

os.environ["WANDB_SILENT"] = "true"

class TensorboardLogger(object):
    """
    Log objects to tensorboard.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add tensorboard CLI args.
        """
        logger = argparser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog',
            '--tensorboard-log',
            type='bool',
            default=False,
            help="Tensorboard logging of metrics, default is %(default)s",
            hidden=False,
        )

    def __init__(self, opt: Opt):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        tbpath = opt['student_model_file'] + '.tensorboard'
        logging.debug(f'Saving tensorboard logs to: {tbpath}')
        if not os.path.exists(tbpath):
            os.makedirs(tbpath)
        self.writer = SummaryWriter(tbpath, comment=json.dumps(opt))

    def log_metrics(self, setting, step, report):
        """
        Add all metrics from tensorboard_metrics opt key.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            if isinstance(v, numbers.Number):
                self.writer.add_scalar(f'{k}/{setting}', v, global_step=step)
            elif isinstance(v, Metric):
                self.writer.add_scalar(f'{k}/{setting}', v.value(), global_step=step)
            else:
                logging.error(f'k {k} v {v} is not a number')

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    """
    Log object for wandb
    """
    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add wandb CLI args.
        """
        logger = argparser.add_argument_group('Wandb Logging Arguments')
        logger.add_argument(
            '--wand-project-name',
            type=str,
            help="Project Name for specific wandb run",
            default='Karu_chatbot_v0',
            hidden=False,
        )
        logger.add_argument(
            '--wand-run-name',
            type=str,
            help="Project Run Name for specific wandb run",
            required=True,
            hidden=False,
        )
        logger.add_argument(
            '--wand-id',
            type=str,
            help="ID for specific run that will used to resume previously preemted logging",
            required=True,
            hidden=False,
        )

    def __init__(self, opt: Opt):
        try:
            import wandb
        except ImportError:
            raise ImportError('Please run `pip install wandb -qqq`.')
        
        # login to wandb
        wandb.login()

        # gather config
        config = {
            'optimizer': opt['optimizer'],
            'embedding_size': opt['student_config']['embedding_size'],
            'ffn_size': opt['student_config']['ffn_size'],
            'n_decoder_layers': opt['student_config']['n_decoder_layers'],
            'tokenizer': opt['dict_tokenizer'],
            'fp16': opt['fp16'],
            'fp16_impl': opt['fp16_impl'],
            'lr': opt['lr'],
            'lr_scheduler': opt['lr_scheduler'],
            }

        wandb.init(
            name= opt['wand_run_name'], resume=True, project=opt['wand_project_name'], 
            id=opt['wand_id'], config=config
        )

        self.key_map = {
            'loss': 'Cross Entropy Loss (token loss)',
            'ppl': 'perplexity (ppl)',
            "token_acc": "Token Accuracy",
            "exps": "examples per sec",
            "ltpb": "label token per batch",
            "ltps": "label token per sec",
            "ctpb": "text token per batch",
            "ctps": "text token per sec",
            "tpb": "total token per batch",
            "tps": "total token per sec",
            "ups": "Update Timer",
            "gnorm": "gradient norm",
            "clip": "gradient clip",
            "fp16_loss_scalar": "Fp16 loss scale",
            "lr": "Learning Rate",
            "gpu_mem": "GPU Mem",
            "total_train_updates": "Total Train updates",
            "total_epochs": "Total Epochs",
            "total_exs": "total Examples",
            "parleys": "Steps",
            "train_time": "Train Time",
        }

    def log_metrics(self, setting, step, report):
        """
        log all metrics to wandb.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        metrics = {'Training Step': step}
        for k, v in report.items():
            if k in self.key_map:
                if isinstance(v, numbers.Number):
                    metrics[f'{self.key_map[k]}/{setting}'] = v
                elif isinstance(v, Metric):
                    metrics[f'{self.key_map[k]}/{setting}'] = v.value()
                else:
                    logging.error(f'k {k} v {v} is not a number')
            else:
                if isinstance(v, numbers.Number):
                    metrics[f'{k}/{setting}'] = v
                elif isinstance(v, Metric):
                    metrics[f'{k}/{setting}'] = v.value()
                else:
                    logging.error(f'k {k} v {v} is not a number')

        wandb.log(metrics)