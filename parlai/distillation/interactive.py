from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging
import os
from parlai.utils.torch import atomic_save
import random
from parlai.gcp.gcs_service import gcp as storage_agent
import traceback
import json
import torch

def get_latest_train(file_path):
    try:
        cand = list(set([ os.path.join(*os.path.split(i)[:1]) for i in storage_agent.list_files(file_path) if os.path.split(i)[1].strip() !='']))
        cand = [i for i in cand if '.tensorboard' not in i ]
        cand = {int(i.split('_')[-1]):i for i in cand}
        latest = sorted(list(cand.keys()), reverse=True)[0]
        latest = cand[latest]
        return latest
    except:
        traceback.print_exc()
        return False

def check_parlai_model(path: str):
    """
    Return opt and model states.

    Override this method for more specific loading.
    """
    import parlai.utils.pickle

    states = torch.load(
        path, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
    )
    if states.get('student_model'):
        states['model'] = states['student_model']
        del states['student_model']
        atomic_save(states, path)

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'Interactive chat with a model on the command line'
        )
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    storage_tag = parser.add_argument_group('tag to fetch model from gcs')
    storage_tag.add_argument(
        '--run-tag',
        type=str,
        help='specifc tag for training run with specific hyper-paramter or model'
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default='',
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Set to the empty string to not save at all',
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    WorldLogger.add_cmdline_args(parser)
    return parser


def interactive(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    human_agent = LocalHumanAgent(opt)
    # set up world logger
    world_logger = WorldLogger(opt) if opt.get('outfile') else None
    world = create_task(opt, [human_agent, agent])

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()
        if world.epoch_done() or world.get_total_parleys() <= 0:
            # chat was reset with [DONE], [EXIT] or EOF
            if world_logger is not None:
                world_logger.reset()
            continue

        if world_logger is not None:
            world_logger.log(world)
        if opt.get('display_examples'):
            print("---")
            print(world.display())

    if world_logger is not None:
        # dump world acts to file
        world_logger.write(opt['outfile'], world, file_format=opt['save_format'])


@register_script('interactive', aliases=['i'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):

        latest_train_path = get_latest_train(self.opt['run_tag'])

        model_download_path = os.path.join(*os.path.split(self.opt['model_file'])[:-1])
        if latest_train_path:
            if len(os.listdir(model_download_path)) < 2 or os.path.isdir(model_download_path):
                storage_agent.download_all(latest_train_path, model_download_path)

        for subdir in os.listdir(model_download_path):
            if subdir.endswith('.checkpoint'):
                model_file_path = subdir    
        
        if os.path.isfile(os.path.join(model_download_path, model_file_path.replace('.checkpoint', ''))):
            model_file_path = model_file_path.replace('.checkpoint', '')
        self.opt['dict_file'] =  os.path.join(model_download_path, model_file_path + '.dict')
        self.opt['model_file'] = os.path.join(model_download_path, model_file_path)
        with open(self.opt['model_file']+'.opt') as fp:
            student_config = json.loads(fp.read())

        student_config.update(student_config['student_config'])
        with open(self.opt['model_file']+'.opt', 'w') as fw:
            json.dump(student_config, fw)
        
        self.opt.update(student_config['student_config'])
        self.opt['embedding_size'] = student_config['student_config']['embedding_size']
        check_parlai_model(self.opt['model_file'])

        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()
