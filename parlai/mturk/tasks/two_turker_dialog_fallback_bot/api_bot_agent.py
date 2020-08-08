import os
import queue
import time
import json
import requests
from parlai.core.agents import Agent


class APIBotAgent(Agent):
    def __init__(self, opt, agent_id, run_id, shared=None):
        super(APIBotAgent, self).__init__(opt=opt, shared=shared)
        self.id = agent_id
        self.bot_url = f'http://{self.opt["bot_host"]}:{str(self.opt["bot_port"])}'
        self.session_id = None
        self.resp_queue = queue.Queue()
        self.bot_response_time = []
        self.bot_request_error_counter = 0
        self.run_id = run_id

    def observe(self, observation):
        if self.session_id:
            observation.update({'session_id': self.session_id})
        try:
            t = time.time()
            response = requests.post(f'{self.bot_url}/interact',
                                     json=observation,
                                     auth=(self.opt['bot_username'],
                                           self.opt['bot_password']))
            self.bot_response_time.append(time.time() - t)
            response = json.loads(response.content)
            response["text"] = response["bot_reply"]["text"]
            del response["bot_reply"]
            if "session_id" in response:
                self.session_id = response["session_id"]
            self.resp_queue.put(response)
        except Exception as e:
            print(repr(e))
            self.bot_request_error_counter += 1

    def act(self):
        result = self.resp_queue.get()
        result.update({'id': self.id,
                       'episode_done': False})
        return result

    def shutdown(self):
        super(APIBotAgent, self).shutdown()
        folder = 'sandbox' if self.opt['is_sandbox'] else 'live'
        time_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'run_data', f'{folder}', f'{self.run_id}', 'bot_response_time.txt')
        error_counter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          'run_data', f'{folder}', f'{self.run_id}', 'bot_error_counter.txt')
        with open(time_log_file, 'a') as f:
            f.write(str(self.bot_response_time) + '\n')

        with open(error_counter_file, 'a') as f:
            f.write(str(self.bot_request_error_counter) + '\n')

        _ = requests.post(f'{self.bot_url}/interact',
                          json={'text': '[DONE]'},
                          auth=(self.opt['bot_username'],
                                self.opt['bot_password']))
