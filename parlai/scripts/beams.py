from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.utils.strings import colorize
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
    WorldLogger.add_cmdline_args(parser)
    return parser

def interactive(opt):
    print(colorize('List of Inference Choice: ','highlight'))
    print(colorize("'greedy', 'topk', 'nucleus', 'delayedbeam'", 'field'))
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    # set up world logger
    world_logger = WorldLogger(opt) if opt.get('outfile') else None

    while True:
        reply_text = input(colorize("Enter Your Message:", 'labels') + ' ')
        if '[DONE]' in reply_text:
            break
        agent.observe({'text':reply_text,'episode_done':False})
        _, beams = agent.act()
        print(colorize('Generated Beams: ','labels'))
        for (val, _) in beams:
            print(val)

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