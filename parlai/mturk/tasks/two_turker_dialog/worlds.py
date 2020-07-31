import random
import numpy as np

from joblib import Parallel, delayed
from parlai.core.worlds import validate
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from .robot_persona_list import robot_personas
from .child_personas import gen_child_persona_sentence


class TwoTurkerDialogOnboardWorld(MTurkOnboardWorld):
    def __init__(self, opt, mturk_agent):
        super(TwoTurkerDialogOnboardWorld, self).__init__(opt, mturk_agent)
        self.opt = opt

    def parley(self):
        self.send_task_instruction(self.get_instruction('no_qualification'))
        self.episodeDone = True

    def send_task_instruction(self, message):
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'qual_test_pass': True,
            'text': message,
            'onboard_message': (
                '<br>'
                '<b><h4>Task Instruction</h4></b>'
                '<br>'
                f'Once you\'re paired with a turker you will be assigned a character. Chat with another worker as if '
                'you\'ve that character.'
            )
        })
        agent_act = self.mturk_agent.act(timeout=self.opt["max_resp_time"])
        if self.check_timeout(agent_act):
            return

    def check_timeout(self, act):
        if act['text'] == TIMEOUT_MESSAGE and act['episode_done']:
            self.episodeDone = True
            return True
        else:
            return False

    def shutdown(self):
        pass

    def expire_hit(self, message=None):
        self.mturk_agent.mturk_manager.force_expire_hit(self.mturk_agent.worker_id,
                                                        self.mturk_agent.assignment_id,
                                                        text=message)

    def get_instruction(self, tag):
        if tag == 'passed_first_time':
            return (
                'Congratulations you completed qualification task.\n'
                'Now, Please read the instructions carefully and <b>when you are ready '
                f'send anything to continue.</b>\n Please respond within {self.opt["max_onboard_resp_time"] // 60} minutes. '
                'We need to pair you with another turker.'
            )
        if tag == 'already_passed':
            return (
                'Welcome back! You\'ve already completed our qualification task. \n'
                'Please read the instruction carefully and <b>when you are ready '
                'send anything to continue.</b>'
                f'\n Please respond within {self.opt["max_onboard_resp_time"] // 60} minutes. '
                'We need to pair you with another turker.'
            )
        if tag == 'no_qualification':
            return (
                'Thank you for accepting the HITs. '
                '<b>Please read the instruction carefully and when you are ready '
                'send anything to continue.</b>.'
            )


class TwoTurkerDialogWorld(MTurkTaskWorld):
    def __init__(self, opt, agents):
        self.opt = opt
        self.agents = agents
        self.episodeDone = False
        self.turn_index = 0
        self.dialog = []
        self.n_turn = np.random.randint(self.opt['range_turn'][0],
                                        self.opt['range_turn'][1]) + 1
        self.assign_conv_role()

    def assign_conv_role(self):
        child_persona_text = gen_child_persona_sentence()
        robot = random.choice(robot_personas)
        robot_persona_text = (
            f'{robot["title"]} Karu. '
            f'{robot["description"]}'
        )
        for agent in self.agents:
            if agent.id == 'CHILD':
                agent.persona_text = child_persona_text
            else:
                agent.persona_text = robot_persona_text

    def parley(self):
        self.turn_index += 1
        acts = {}

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}

        """If at first turn, we need to give each agent some prior info if any like personas"""
        if self.turn_index == 1:
            for agent in self.agents:
                control_msg['text'] = self.get_instruction(agent=agent, tag='start', )
                control_msg['show_persona'] = True
                control_msg['persona_description'] = (
                    '<br>'
                    '<b><h3>Your Persona</h3></b>'
                    '<br>'
                    f"You're assigned with the following character: <br>"
                    f'<b><span style="color:blue">{agent.persona_text}</span></b>'
                    '<br>'
                )
                agent.observe(validate(control_msg))

        """If we get to the min turns, inform turker that they can end if they want"""
        if self.turn_index == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """If max turns reached, end the episode"""
        if self.turn_index == self.opt['max_turns'] + 1:
            for agent in self.agents:
                control_msg['text'] = self.get_instruction(None, 'exceed_max_turns')
                control_msg['episode_done'] = True
                agent.observe(validate(control_msg))
            self.episodeDone = True
            return

        """Otherwise, we proceed accordingly"""
        for agent in self.agents:
            if not self.episodeDone:
                acts[agent.id] = agent.act(
                    timeout=self.opt['max_resp_time'] + (60 if self.turn_index == 1 else 0))  # More time for 1st turn
            if self.check_timeout(acts[agent.id]):
                return

            if self.turn_index > 1:
                # only check if message is too short except on first message
                while self.is_msg_tooshortlong(acts[agent.id], agent):
                    acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'])
                    if self.check_timeout(acts[agent.id]):
                        return

            if acts[agent.id]['episode_done']:
                self.episodeDone = True
                for ag in self.agents:
                    # if agent disconnected
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = (
                            'The other worker unexpectedly disconnected. '
                            'Please click "Done with this HIT" button below to '
                            'finish this HIT.'
                        )
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                        return
                # agent ends chat after exceeding minimum number of turns
                if self.turn_index > self.n_turn:
                    for ag in self.agents:
                        if ag != agent:
                            ag.observe(validate(acts[agent.id]))
                        control_msg['text'] = (
                            'One of you ended the chat. Thanks for your time! '
                            'Please click "Done with this HIT" button below '
                            'to finish this HIT.'
                        )
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                return

            while (self.turn_index <= self.n_turn) and acts[agent.id]['episode_done']:
                control_msg['text'] = self.get_instruction(agent=agent, tag='chat_not_done')
                control_msg['episode_done'] = False
                agent.observe(validate(control_msg))
                acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'])
                if self.check_timeout(acts[agent.id]):
                    return

            else:
                self.dialog.append({"turn_index": self.turn_index,
                                    "id": agent.id,
                                    "text": acts[agent.id]["text"]})
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[agent.id]))

    def get_instruction(self, agent=None, tag='first', agent_id=None):
        if tag == 'start':
            return (
                    '\nSuccessfully matched. Now let\'s get to know each other '
                    'through the chat! \nYou need to finish at least <b>'
                    + str(self.n_turn)
                    + ' chat turns</b>, after that you can click the "Done" button '
                      'to end the chat.\n'
                      '<b>You\'re assigned with following persona:<b>'
                      f'<br><b><span style="color:blue">{agent.persona_text}</span></b><br>'
                      '<b>You can also track the character description on the left.</b> '
                      '\n <span style="color:blue"><b>Please try to speak to the '
                      'other person as if you\'re the character mentioned .</b></span>'
                      '\n <span style="color:blue"><b>Do not trivially copy the '
                      'character descriptions into the message.</b></span> \n'
                      'Please respond quickly. We need it interactive and real time.'
            )

        if tag == 'chat_not_done':
            return (
                    'Sorry, we need at least <b>'
                    + str(self.n_turn + 1 - self.turn_index)
                    + ' more turn(s)</b> to finish. '
                      'Please send a new message:'
            )

        if tag == 'timeout':
            return (
                '<b>{}</b> is timeout. Please click the "Done with this HIT" '
                'button below to exit this HIT. No rejections.'.format(agent_id)
            )

        if tag == 'exceed_min_turns':
            return (
                '\n {} chat turns finished! \n Keep chatting or you can click '
                'the "Done" button to end the chat if it\'s your turn.'.format(
                    self.n_turn
                )
            )
        if tag == 'exceed_max_turns':
            return (
                f'\n {self.opt["max_turns"]} chat turns finished! \n We cannot support more than this. \n'
                'Please click the "Done with this HIT" '
                'button below to finish this HIT.'
            )

    def is_msg_tooshortlong(self, act, ag):
        th_min = self.opt['min_num_words_in_message']
        th_max = self.opt['max_num_words_in_message']
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}

        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = (
                'Your message is too short, please make it more than '
                f'<b><span style="color:red">{th_min} words</span></b>.'
            )
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = (
                'Your message is too long, please make it less than '
                f'<b><span style="color:red">{th_max} words</span></b>.'
            )
            ag.observe(validate(control_msg))
            return True
        return False

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True,
                           'id': 'SYSTEM',
                           'text': self.get_instruction(
                               agent_id=act['id'], tag='timeout'
                           )}
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.episodeDone = True
            return True
        else:
            return False

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
        if self.opt.get('immediate_assignment_approve'):
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
        return {
            'personas': {ag.id: ag.persona_text for ag in self.agents},
            'conversations': self.dialog
        }
