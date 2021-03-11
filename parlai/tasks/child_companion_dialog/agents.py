import copy
import os

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build


class ChildCompanionDialogTeacher(ParlAIDialogTeacher):
    @classmethod
    def add_cmdline_args(cls, parser):
        parser = parser.add_argument_group('CCD opt')
        parser.add_argument(
            '--task-data-version',
            type=str,
            default='All',
            help="Specify which version of data to use"
        )
        parser.add_argument(
            '--min-dialogue-turns',
            type=int,
            default=-1,
            help="Minimum number of turns required in conversation to be considered in dataset "
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)
        super().__init__(opt, shared)


class DefaultTeacher(ChildCompanionDialogTeacher):
    pass


def _processed_data_path(opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'child_companion_dialog', opt.get('task_data_version', 'All'), dt + '.txt')
