import copy
import os

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build

class ChildCompanionDialogTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        dt = opt['datatype'].split(':')[0]
        opt['parlaidialogteacher_datafile'] = os.path.join(opt['datapath'], 'unliked_utterances', f'{dt}.txt')
        super().__init__(opt, shared)


class DefaultTeacher(ChildCompanionDialogTeacher):
    pass