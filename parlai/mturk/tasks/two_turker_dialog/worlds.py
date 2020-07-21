from joblib import Parallel, delayed
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld


class TwoTurkerDialogOnboardWorld(MTurkOnboardWorld):

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


class TwoTurkerDialogWorld(MTurkTaskWorld):
    def __init__(self, opt, agents):
        self.agents = agents
        self.episodeDone = False
        self.turn_index = 0
        self.dialog = []

    def parley(self):
        self.turn_index += 1
        acts = {}
        if self.turn_index == 1:
            for agent in self.agents:
                agent.observe({
                    'id': 'SYSTEM',
                    'text': (
                        'You are paired with a Turker, Have really good conversation..'
                        'Only send message in your turn.'
                    )
                })
        for agent in self.agents:
            acts[agent.id] = agent.act(timeout=None)
            if acts[agent.id]['episode_done']:
                self.episodeDone = True
                for ag in self.agents:
                    ag.observe(validate({
                        'id': 'SYSTEM',
                        'text': (
                            'One of you ended the chat'
                            'Please click "Done with this HIT" button below to'
                            'finish this HIT.'
                        ),
                        'episode_done': True
                    }))
                return
            else:
                self.dialog.append({"turn_index": self.turn_index,
                                    "id": agent.id,
                                    "text": acts[agent.id]["text"]})
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[agent.id]))

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )

    def review_work(self):
        global review_agent

        def review_agent(ag):
            if hasattr(ag, 'not_approve'):
                pass
            else:
                ag.approve_work()

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(review_agent)(agent) for agent in self.agents
        )

    def get_custom_task_data(self):
        return {'conversations': self.dialog}
