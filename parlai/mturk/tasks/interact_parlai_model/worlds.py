from joblib import Parallel, delayed
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld


class InteractParlAIModelOnboardWorld(MTurkOnboardWorld):

    def parley(self):
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'text': (
                'Welcome! If you are ready,'
                'please click "I am ready, continue" to start this task.'
            ),
            # This is message(Can be formated in html) that can be shown in onboarding page
        })
        self.mturk_agent.act()
        self.episodeDone = True


class InteractParlAIModelWorld(MTurkTaskWorld):
    def __init__(self, opt, mturk_agent, parlai_agent):
        self.mturk_agent = mturk_agent
        self.parlai_agent = parlai_agent
        self.episodeDone = False
        self.turn_index = 0
        self.dialog = []

    def parley(self):
        self.turn_index += 1
        acts = {}
        if self.turn_index == 1:
            self.mturk_agent.observe({
                'id': 'SYSTEM',
                'text': (
                    'You can talk with the bot now.'
                    'Only send message in your turn.'
                )
            })
        for agent in [self.mturk_agent, self.parlai_agent]:
            if agent == self.mturk_agent:
                acts[agent.id] = agent.act(timeout=None)  # TODO add agent timeout for action
            else:
                acts[agent.id] = agent.act()
            if acts[agent.id]['episode_done']:
                self.episodeDone = True

                self.mturk_agent.observe(validate({
                    'id': 'SYSTEM',
                    'text': (
                        'Thanks for conversing with our bot.'
                    ),
                    'episode_done': True
                }))
                return
            else:
                self.dialog.append({"turn_index": self.turn_index,
                                    "id": agent.id,
                                    "text": acts[agent.id]["text"]})
                for other_agent in [self.mturk_agent, self.parlai_agent]:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[agent.id]))

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        self.mturk_agent.shutdown()
        self.parlai_agent.shutdown()

    def review_work(self):
        if hasattr(self.mturk_agent, 'not_approve'):
            pass
        else:
            self.mturk_agent.approve_work()

    def get_custom_task_data(self):
        return {'conversations': self.dialog}
