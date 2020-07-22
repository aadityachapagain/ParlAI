from joblib import Parallel, delayed
from parlai.core.worlds import validate
from parlai.mturk.core import mturk_utils
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld


class TwoTurkerDialogOnboardWorld(MTurkOnboardWorld):
    def __init__(self, opt, mturk_agent, pass_qual_id, fail_qual_id):
        super(TwoTurkerDialogOnboardWorld, self).__init__(opt, mturk_agent)
        self.opt = opt
        self.pass_qual_id = pass_qual_id
        self.fail_qual_id = fail_qual_id
        self.first_time = not mturk_utils.check_worker_qualification_exist(
            self.pass_qual_id,
            self.mturk_agent.worker_id,
            is_sandbox=self.opt['is_sandbox']
        )
        self.pass_qual_test = False if self.first_time else True

    def parley(self):
        if self.first_time:
            # TODO Present worker the qualification task
            self.mturk_agent.observe({
                'id': 'SYSTEM',
                'text': (
                    'You\'re here for the first time. We need you to complete a '
                    'qualification test before you proceed to the main task.'
                )
            })
            self.mturk_agent.observe({
                'id': 'SYSTEM',
                'text': 'What is 1 + 1? \n Please give an integer result.'
            })
            agent_act = self.mturk_agent.act()
            if int(agent_act['text']) == 2:
                self.pass_qual_test = True
                self.mturk_agent.observe({
                    'id': 'SYSTEM',
                    'text': 'Congratulations you completed qualification task. You may proceed to main task'
                })
            else:
                self.pass_qual_test = False
                self.mturk_agent.observe({
                    'id': 'SYSTEM',
                    'text': 'Sorry you\'ve failed our qualification test task. You cannot proceed to main task.'
                })
        else:
            # Worker is good to go to main task
            self.mturk_agent.observe({
                'id': 'SYSTEM',
                'text': (
                    'Welcome back! You\'ve already completed our qualification task. '
                    'You\'re good to go. If you are ready, '
                    'Please send anything to continue.'
                ),
            })
            self.mturk_agent.act()
        self.episodeDone = True

    def shutdown(self):
        if self.pass_qual_test:
            if self.first_time:
                mturk_utils.give_worker_qualification(
                    self.mturk_agent.worker_id,
                    self.pass_qual_id,
                    is_sandbox=self.opt['is_sandbox']
                )
        else:
            mturk_utils.give_worker_qualification(
                self.mturk_agent.worker_id,
                self.fail_qual_id,
                is_sandbox=self.opt['is_sandbox']
            )
            self.mturk_agent.shutdown()
            self.mturk_agent.reject_work(reason="Turker failed the qualification task")


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
        # TODO need to disable auto approve if manual review is done
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
