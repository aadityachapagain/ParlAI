import os
import queue
import time
import logging
from parlai.core.agents import Agent
import parlai.mturk.core.shared_utils as shared_utils
from parlai.core.agents import create_agent_from_shared
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


class BotAgent(Agent):
    def __init__(self, opt, agent, agent_id, run_id, offensive_language_classifier=None, shared=None):
        super(BotAgent, self).__init__(opt=opt, shared=shared)
        self.id = agent_id
        self.agent = create_agent_from_shared(agent.share())
        self.resp_queue = queue.Queue()
        self.bot_response_time = []
        self.bot_request_error_counter = 0
        self.run_id = run_id
        self._init_safety(opt, offensive_language_classifier)

    def _init_safety(self, opt, offensive_language_classifier):
        """
        Initialize safety modules.
        """
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            self.offensive_string_matcher = OffensiveStringMatcher()
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            if offensive_language_classifier:
                self.offensive_classifier = OffensiveLanguageClassifier(shared=offensive_language_classifier.share())
            else:
                self.offensive_classifier = OffensiveLanguageClassifier()

        self.self_offensive = False

    def check_offensive(self, text):
        """
        Check if text is offensive using string matcher and classifier.
        """
        if text == '':
            return False
        if (
            hasattr(self, 'offensive_string_matcher')
            and text in self.offensive_string_matcher
        ):
            return True
        if hasattr(self, 'offensive_classifier') and text in self.offensive_classifier:
            return True

        return False

    def observe(self, observation):
        shared_utils.print_and_log(logging.INFO,
                                   f"Sending {observation} to bot......")
        try:
            t = time.time()
            self.agent.observe(observation)
            if not observation.get('persona'):
                response = self.agent.act()
            else:
                response = {
                        'text': "Successfully set Persona",
                        'episode_done': False
                    }
            self.bot_response_time.append(time.time() - t)
            self.resp_queue.put(response)
            shared_utils.print_and_log(logging.INFO,
                                       f"Received {response} from bot.......")
        except Exception as e:
            print(repr(e))
            self.bot_request_error_counter += 1

    def act(self):
        result = self.resp_queue.get()
        result.update({'id': self.id})
        return result

    def shutdown(self):
        super(BotAgent, self).shutdown()
        self.agent.shutdown()

        folder = 'sandbox' if self.opt['is_sandbox'] else 'live'
        time_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'run_data', f'{folder}', f'{self.run_id}', 'bot_response_time.txt')
        error_counter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          'run_data', f'{folder}', f'{self.run_id}', 'bot_error_counter.txt')
        with open(time_log_file, 'a') as f:
            f.write(str(self.bot_response_time) + '\n')

        with open(error_counter_file, 'a') as f:
            f.write(str(self.bot_request_error_counter) + '\n')
