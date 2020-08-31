from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build, matcher
import parlai.utils.logging as logging
from parlai.utils.misc import str_to_msg
import random

import os
from typing import List, Tuple


class RedditTeacher(DialogTeacher):
    """
    Reads reddit datasets
    """

    def __init__(self, opt, shared=None):
        opt['task'] = 'reddit_datasets:chunks'
        build(opt)
        self.opt = opt
        self.datasets_type = 'train' if 'train' in opt.get('datatype','train') else 'valid'
        opt['datafile'] = os.path.join(
            opt['datapath'], 'reddit_datasets/train_data'
        )
        self.id = 'reddit_datasets'
        super().__init__(opt, shared)

    def setup_data(self, path):
        req_files = random.shuffle([ '{}-0000{}-of-00005.txt'.format(self.datasets_type,i) for i in range(5)])
        if self.datasets_type == 'valid':
            req_files = random.sample(req_files, random.randint(1,2))
        for subdir in req_files:
            subdir_path = os.path.join(path, subdir)
            with open(subdir_path, newline='\n', encoding="utf-8") as read:
                for line_no, line in enumerate(read, 1):
                    msg = str_to_msg(line.rstrip('\n'))
                    if msg and 'eval_labels' in msg:
                        raise ValueError(
                            f"It looks like you've written eval_labels as a key in your "
                            f"data file. This is not appropriate; labels will be converted "
                            f"for you automatically. This is happening on Line {line_no} "
                            f"in {path}. The line is:\n\t{line}"
                        )
                    if msg and 'text' not in msg:
                        raise ValueError(
                            f'ParlaiDialogTeacher requires a "text" field in every '
                            f'entry, but one is missing in Line {line_no} in {path}. '
                            f'The line is:\n\t{line}'
                        )
                    if msg and 'labels' not in msg:
                        raise ValueError(
                            f'ParlaiDialogTeacher requires a "labels" field in every '
                            f'entry, but one is missing in Line {line_no} in {path}. '
                            f'The line is:\n\t{line}'
                        )
                    if msg:
                        episode_done = msg.get('episode_done', False)
                        yield (msg['text'], msg['labels']), episode_done


class DefaultTeacher(RedditTeacher):
    pass
