import queue
import json
import requests
from parlai.core.agents import Agent


class APIBotAgent(Agent):
    def __init__(self, opt, agent_id, shared=None):
        super(APIBotAgent, self).__init__(opt=opt, shared=shared)
        self.id = agent_id
        self.bot_url = f'http://{self.opt["bot_host"]}:{str(self.opt["bot_port"])}'
        self.session_id = None
        self.resp_queue = queue.Queue()
        self.authenticate_agent()

    def authenticate_agent(self):
        auth_resp = requests.post(f'{self.bot_url}/token',
                                  data={'username': self.opt['bot_username'],
                                        'password': self.opt['bot_password']})
        self.auth_token = json.loads(auth_resp.content)['access_token']
        print("Successfully authenticated to BOT API")

    def get_headers(self):
        return {'Authorization': 'Bearer ' + self.auth_token}

    def observe(self, observation):
        if self.session_id:
            observation.update({'session_id': self.session_id})
        response = requests.post(f'{self.bot_url}/interact',
                                 json=observation,
                                 headers=self.get_headers())
        response = json.loads(response.content)
        if "session_id" in response:
            self.session_id = response["session_id"]
        self.resp_queue.put(response)

    def act(self):
        result = self.resp_queue.get()
        result["text"] = result["bot_reply"]["text"]
        del result["bot_reply"]
        result.update({'id': self.id,
                       'episode_done': False})
        return result
