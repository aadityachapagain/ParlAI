from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging
import os
import random
import time

generated_adult_beams = []
INFERENCE_FLAG = True
AT_CHECKPOINT = False

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
    parser.add_argument(
        '--adultlike_file_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--recursion',
        default=3,
        type=int
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

def _generate_adult_sentences(agent, statement,curr_gen, max_gen = 3):
    agent.observe({'text': statement, 'episode_done': True})
    start = time.time()
    _, beams = agent.act()
    ttl = time.time() - start
    agent.reset()
    if isinstance(beams, list) and len(beams) > 1 and curr_gen <= max_gen:
        for beam, score in beams:
            generated_adult_beams.append(f'text:{statement}\tlabels:{beam}\tepisode_done:True\treward:-1\tgen:{str(curr_gen)}\tttl:{str(ttl)}\tscore:{str(score)}')
            print(f'text:{statement}\tlabels:{beam}\tepisode_done:True\treward:-1\tgen:{str(curr_gen)}\tttl:{str(ttl)}\tscore:{str(score)}')
            _generate_adult_sentences(agent, beam, curr_gen + 1, max_gen)
    

def _emtpy_generated_list():
    global generated_adult_beams
    generated_adult_beams = []


def _save_file_disk(inf):
    with open(f'data/generated_adult_beams_{inf}.txt', 'a+') as fw:
        for stmt in generated_adult_beams:
            fw.write(stmt+'\n')

def _write_checkpoint(val, inf):
    with open(f'data/generated_adult_beams_{inf}.checkpoint', 'w') as fw:
        fw.write(str(val))

def _load_checkpoint(inf):
    try:
        with open(f'data/generated_adult_beams_{inf}.checkpoint', 'r') as fw:
            val = int(fw.read())
    except:
        val = 0
    return val

def interactive(opt):
    inference = opt.get('inference')
    file_path = opt.get('adultlike_file_path')
    recursion = opt.get('recursion')
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    human_agent = LocalHumanAgent(opt)
    # set up world logger
    world_logger = WorldLogger(opt) if opt.get('outfile') else None
    adult_sentences = []
    global AT_CHECKPOINT
    with open(file_path,  'r') as fd:
        adult_sentences =  [i.strip() for i in fd]
    
    checkpoint = _load_checkpoint(inference)
    for idx, statement in enumerate(adult_sentences):
        if idx >= checkpoint:
            AT_CHECKPOINT = True
        else:
            continue
        _generate_adult_sentences(agent, statement, 0, recursion)
        _write_checkpoint(idx, inference)
        _save_file_disk(inference)
        _emtpy_generated_list()
@register_script('interactive', aliases=['i'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()