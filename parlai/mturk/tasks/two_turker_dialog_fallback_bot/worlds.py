import numpy as np
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkTaskWorld
from parlai.mturk.tasks.two_turker_dialog.child_personas import gen_child_persona_sentence
from parlai.mturk.tasks.two_turker_dialog.worlds import TwoTurkerDialogOnboardWorld


class TwoTurkerDialogFallbackBotOnboardWorld(TwoTurkerDialogOnboardWorld):
    def send_task_instruction(self, message):
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'qual_test_pass': True,
            'text': message,
            'onboard_message': (
                '<b><h4>Task Instruction</h4></b>'
                f'Once you\'re paired with a turker you will be assigned a character. Chat with another worker as if '
                'you\'ve that character. If not able to pair you\'ll chat with Karu playing a child role with assigned character.'
            )
        })
        agent_act = self.mturk_agent.act(timeout=self.opt['max_onboard_resp_time'])
        if self.check_timeout(agent_act):
            return

    def get_instruction(self, tag):
        if tag == 'passed_first_time':
            return (
                'Congratulations you completed qualification task.\n'
                'Now, Please read the instructions carefully and <b>when you are ready '
                f'send anything to continue.</b>\n Please respond within {self.opt["max_onboard_resp_time"] // 60} minutes. '
                f'We need to pair you with another turker. If not able to pair within {self.opt["waiting_pool_time"]//60} minutes after your response, you\'ll be paired with Karu'
            )
        if tag == 'already_passed':
            return (
                'Welcome back! You\'ve already completed our qualification task. \n'
                'Please read the instruction carefully and <b>when you are ready '
                'send anything to continue.</b>'
                f'\n Please respond within {self.opt["max_onboard_resp_time"] // 60} minutes. '
                f'We need to pair you with another turker. If not able to pair within {self.opt["waiting_pool_time"]//60} minutes after your response, you\'ll be paired with Karu'
            )


class InteractParlAIModelWorld(MTurkTaskWorld):
    def __init__(self, opt, mturk_agent, parlai_agent):
        self.opt = opt
        self.mturk_agent = mturk_agent
        self.parlai_agent = parlai_agent
        self.episodeDone = False
        self.turn_index = 0
        self.dialog = []
        self.n_turn = np.random.randint(self.opt['range_turn'][0],
                                        self.opt['range_turn'][1]) + 1
        self.assign_conv_role()
        self.bot_eval_by_worker = None

    def assign_conv_role(self):
        child_persona_text = gen_child_persona_sentence()
        self.mturk_agent.persona_text = child_persona_text

    def parley(self):
        self.turn_index += 1
        acts = {}

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}
        """If at first turn, we need to give each agent some prior info if any like personas"""
        if self.turn_index == 1:
            control_msg['text'] = self.get_instruction(tag='start')
            control_msg['show_persona'] = True
            control_msg['persona_description'] = (
                '<br>'
                '<b><h3>Your Persona</h3></b>'
                f"You're assigned with the following character: <br>"
                f'<b><span style="color:blue">{self.mturk_agent.persona_text}</span></b>'
                '<br>'
            )
            self.mturk_agent.observe(validate(control_msg))

        """If we get to the min turns, inform turker that they can end if they want"""
        if self.turn_index == self.n_turn + 1:
            control_msg['text'] = self.get_instruction(tag='exceed_min_turns')
            control_msg['exceed_min_turns'] = True
            self.mturk_agent.observe(validate(control_msg))

        """If max turns reached, end the episode"""
        if self.turn_index == self.opt['max_turns'] + 1:
            control_msg['text'] = self.get_instruction('exceed_max_turns')
            control_msg['episode_done'] = True
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True
            return

        """Otherwise, we proceed accordingly"""
        for agent in [self.mturk_agent, self.parlai_agent]:
            if agent == self.mturk_agent:
                acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'] + (60 if self.turn_index == 1 else 0))
                if self.check_timeout(acts[agent.id]):
                    return
            else:
                acts[agent.id] = agent.act()

            if self.turn_index > 1 and agent == self.mturk_agent:
                # only check if message is too short/long except on first message
                while self.is_msg_tooshortlong(acts[agent.id], agent):
                    acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'])
                    if self.check_timeout(acts[agent.id]):
                        return
            if acts[agent.id]['episode_done']:
                if (not self.mturk_agent.disconnected) and self.opt.get('bot_evaluation'):
                    self.mturk_agent.observe(validate({
                        'id': 'SYSTEM',
                        'text': self.get_instruction('eval')
                    }))
                eval_act = self.mturk_agent.act(timeout=self.opt['max_resp_time'])
                while eval_act['text'].strip() not in ['1', '2', '3', '4', '5']:
                    control_msg['text'] = self.get_instruction('eval_warn')
                    self.mturk_agent.observe(validate(control_msg))
                    eval_act = self.mturk_agent.act(timeout=self.opt['max_resp_time'])

                self.bot_eval_by_worker = eval_act['text']

                self.mturk_agent.observe(validate({
                    'id': 'SYSTEM',
                    'text': self.get_instruction('end'),
                    'episode_done': True
                }))
                self.episodeDone = True
                return
            else:
                self.dialog.append({"turn_index": self.turn_index,
                                    "id": agent.id,
                                    "text": acts[agent.id]["text"]})
                for other_agent in [self.mturk_agent, self.parlai_agent]:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[agent.id]))

    def get_instruction(self, tag):
        if tag == 'start':
            return (
                    '\nSorry you had to wait too long and we couldn\'t find other worker to pair with you.'
                    'Now you can chat with our existing Karu. \nYou need to finish at least <b>'
                    + str(self.n_turn)
                    + ' chat turns</b>, after that you can click the "Done" button '
                      'to end the chat. Then you can share your experience conversing with Karu. \n'
                      '<b>You can track the character description on the left.</b> '
                      '\n <span style="color:blue"><b>Please try to speak to Karu '
                      ' as if you\'re the character mentioned .</b></span>'
                      '\n <span style="color:blue"><b>Do not trivially copy the '
                      'character descriptions into the message.</b></span> \n'
                      'Please respond quickly. We need it interactive and real time.'
            )
        if tag == 'end':
            return 'Thanks for conversing with our bot.'
        if tag == 'eval':
            return 'How would you <b>rate the conversation with Karu in scale of 1 to 5</b>?'
        if tag == 'eval_warn':
            return 'Please rate within the <b>scale of 1 to 5.</b>'
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
            self.episodeDone = True
            return True
        else:
            return False

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        self.mturk_agent.shutdown()
        self.parlai_agent.shutdown()

    def review_work(self):
        if self.opt.get('immediate_assignment_approve'):
            if hasattr(self.mturk_agent, 'not_approve'):
                pass
            else:
                self.mturk_agent.approve_work()

    def get_custom_task_data(self):
        return {'conversations': self.dialog,
                'worker_persona': self.mturk_agent.persona_text,
                'bot_eval_by_worker': self.bot_eval_by_worker}
