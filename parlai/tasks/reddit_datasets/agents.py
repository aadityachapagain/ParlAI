from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build, matcher
import parlai.utils.logging as logging
from parlai.utils.misc import str_to_msg, warn_once
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
        self.visited = self.get_visited_chunk(opt)
                    
        super().__init__(opt, shared)

    def get_visited_chunk(self, opt):
        visited = []
        if os.path.isfile(os.path.join(opt['datapath'],'reddit_datasets','status.build' )):
            with open(os.path.join(opt['datapath'],'reddit_datasets','status.build' )) as fp:
                for line in fp:
                    visited.append(line.replace('\n', '').strip())

        return visited
    
    def set_visited_chunk(self, chunk):
        with open(os.path.join(self.opt['datapath'],'reddit_datasets','status.build' ), 'a+') as fw:
            fw.write(chunk+ '\n')

    def epoch_end_cleanup(self):
        self.visited = []
        os.remove(os.path.join(self.opt['datapath'],'reddit_datasets','status.build' ))

    def setup_data(self, path):
        req_files = ['{}-00{:03}-of-00200.txt'.format(self.datasets_type, i) for i in range(200)]
        if 'ordered' not in self.opt['datatype']:
            req_files = random.sample(req_files, len(req_files))
        for subdir in req_files:
            subdir_path = os.path.join(path, subdir)
            if (subdir_path in self.visited) and ('ordered' not in self.opt['datatype']):
                logging.log(f'Model already Trained on file {subdir_path}, Skipping batch when not ordered!')
                continue
            # storing batch progression in file to make sure it will not loaded next time
            if 'ordered' not in self.opt['datatype']:
                self.set_visited_chunk(subdir_path)
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
                        warn_once(
                            f'ParlaiDialogTeacher requires a "text" field in every ' +
                            f'entry, but one is missing in Line {line_no} in {path}. ' +
                            f'The line is:\n\t{line}'
                        )
                        continue
                    if msg and 'labels' not in msg:
                        warn_once(
                            f'ParlaiDialogTeacher requires a "labels" field in every ' +
                            f'entry, but one is missing in Line {line_no} in {path}. ' +
                            f'The line is:\n\t{line}'
                        )
                        continue
                    if msg:
                        episode_done = msg.get('episode_done', False)
                        return_msg = {
                            'text': msg['text'],
                            'labels': msg['labels']
                        }
                        if 'subreddit' in msg:
                            return_msg['subreddit'] = msg['subreddit']
                        yield return_msg, episode_done
        # clean up at the end of epoch
        if 'ordered' not in self.opt['datatype']:
            self.epoch_end_cleanup()

class RedditChunkTeacher(ChunkTeacher):
    """
    Full Wikipedia teacher that splits the chunks into train/valid/test.
    """

    def __init__(self, opt, shared=None):
        self.TRAINSIZE = 671513380
        self.VALIDSIZE = 471052
        self.datasets_type = 'train' if 'train' in opt.get('datatype','train') else 'valid'
        self.opt = opt
        if shared is None:
            # set map
            self._set_chunk_idx_to_file()
        else:
            self._set_chunk_idx_to_file()
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return os.path.join(self.opt['datapath'], 'reddit_datasets/train_data')

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt['datatype']
        if 'train' in datatype:
            return self.TRAINSIZE, self.TRAINSIZE
        else:
            # test
            return self.VALIDSIZE, self.VALIDSIZE

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        req_files = random.sample([ '{}-00{:03}-of-00010.txt'.format(self.datasets_type,i) for i in range(200)], 200)
        if self.datasets_type == 'valid':
            req_files = random.sample(req_files,5)
        self.chunk_idx_to_file = {i: x for i, x in enumerate(req_files)}

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        return list(self.chunk_idx_to_file.keys())

    def load_from_chunk(self, chunk_idx: int):
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        output = []
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])
        with open(chunk_path, newline='\n', encoding="utf-8") as read:
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
                    output.append((msg['text'], msg['labels'], episode_done))
        return output

    def create_message(self, queue_output: ChunkOutput, entry_idx=0) -> 'Message':
        """
        Given the tuple output of the queue, return an act.
        """
        text, labels, episode_done = queue_output
        return Message(
            {'text': text, 'labels': labels, 'episode_done': episode_done}
        )

    def share(self):
        shared = super().share()
        shared['chunk_idx_to_file'] = self.chunk_idx_to_file
        return shared
class DefaultTeacher(RedditTeacher):
    pass
