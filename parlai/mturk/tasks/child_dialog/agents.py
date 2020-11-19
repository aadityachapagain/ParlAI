import re
import os
import queue
import time
import logging
import json
import requests
from parlai.core.agents import Agent
import parlai.mturk.core.shared_utils as shared_utils


class BotAgent(Agent):
    def __init__(self, opt, agent_id, run_id, shared=None):
        super(BotAgent, self).__init__(opt=opt, shared=shared)
        self.id = agent_id
        self.session_id = None
        self.resp_queue = queue.Queue()
        self.bot_response_time = []
        self.bot_request_error_counter = 0
        self.run_id = run_id

    def observe(self, observation):
        if observation.get('persona', False):
            self.context_data = [{'context_type': "prime",
                                  'text': sentence.strip()} for sentence in observation['text'].split('\\n')]
            self.resp_queue.put({
                'remote_chat_response': {'output': {'text': "Successfully set persona:"}}
            })
            return

        bot_request_data = {
            "remote_chat_request": {
                "speech": observation['text']
            } if "command" not in observation else {
                "command": observation['command']
            }
        }
        if self.session_id:
            bot_request_data.update({'session_id': self.session_id})
        else:
            # It is a new session; add context data
            # bot_request_data['remote_chat_request'].update({
            #     'extra_lines': self.context_data
            # })
            pass
        try:
            shared_utils.print_and_log(logging.INFO,
                                       f"Sending {observation} to bot......")
            t = time.time()
            response = requests.get(f'http://{self.opt["bot_host"]}:{str(self.opt.get("bot_port", 80))}/interact',
                                    json=bot_request_data,
                                    auth=(self.opt['bot_username'],
                                          self.opt['bot_password']))
            self.bot_response_time.append(time.time() - t)
            response = json.loads(response.content)
            if "session_id" in response:
                self.session_id = response["session_id"]
            self.resp_queue.put(response)
            shared_utils.print_and_log(logging.INFO,
                                       f"Received {response} from bot.......")
        except Exception as e:
            print(repr(e))
            self.bot_request_error_counter += 1

    def act(self):
        bot_response = self.resp_queue.get()
        response = {'id': self.id,
                    'text': re.sub("[\(\[].*?[\)\]]", "", bot_response['remote_chat_response'][
                        'output']['text']),
                    'episode_done': False,
                    'new_session_reason': bot_response['new_session_reason']}
        return response

    def shutdown(self):
        super(BotAgent, self).shutdown()
        folder = 'sandbox' if self.opt['is_sandbox'] else 'live'
        time_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'run_data', f'{folder}', f'{self.run_id}', 'bot_response_time.txt')
        error_counter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          'run_data', f'{folder}', f'{self.run_id}', 'bot_error_counter.txt')
        with open(time_log_file, 'a') as f:
            f.write(str(self.bot_response_time) + '\n')

        with open(error_counter_file, 'a') as f:
            f.write(str(self.bot_request_error_counter) + '\n')
