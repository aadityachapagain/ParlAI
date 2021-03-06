import copy
import os

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build


class ChildCompanionDialogTeacher(ParlAIDialogTeacher):
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
    return os.path.join(opt['datapath'], 'child_companion_dialog', dt + '.txt')
