from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging
import os
import random
import threading

generated_adult_beams = []
INFERENCE_FLAG = True

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

def _generate_adult_sentences(agent, statement, depth = 4):
    depth = depth -1
    generated_adult_beams.append(statement)
    print('statement : ', statement)
    agent.observe({'text': statement, 'episode_done': True})
    beams = agent.act()
    agent.reset()
    if isinstance(beams, list) and len(beams) > 1 and depth > 0:
        for beam in beams:
            _generate_adult_sentences(agent, beam, depth)
    

def save_file_disk():
    with open('data/generated_adult_beams.txt', 'w+') as fw:
        while INFERENCE_FLAG or len(generated_adult_beams) > 0:
            if len(generated_adult_beams):
                fw.write(generated_adult_beams.pop() +'\n')


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
    adult_sentences = []
    with open('data/adult_statements/adult_like_statements.txt',  'r') as fd:
        adult_sentences =  [i.strip() for i in fd]

    for statement in adult_sentences:
        _generate_adult_sentences(agent, statement, 3)
    
    with open('data/generated_adult_beams.txt', 'w+') as fw:
        for stmt in generated_adult_beams:
            fw.write(stmt+'\n')
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