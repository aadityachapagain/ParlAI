import random

import numpy as np
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkTaskWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE, RETURN_MESSAGE, MTURK_DISCONNECT_MESSAGE
from parlai.mturk.tasks.two_turker_dialog.child_personas import gen_child_persona_sentence
from parlai.mturk.tasks.two_turker_dialog.robot_persona_list import robot_personas
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.one_sided_acute_eval_questions import ACUTE_EVAL_QUESTIONS
from parlai.mturk.tasks.two_turker_dialog.worlds import TwoTurkerDialogOnboardWorld


class TwoTurkerDialogFallbackBotOnboardWorld(TwoTurkerDialogOnboardWorld):
    def send_task_instruction(self, message):
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'qual_test_pass': True,
            'text': message,
            'onboard_message': (
                '<b><h4>Task Instruction</h4></b>'
                '<br>'
                f'Once the task starts, you will be given a persona for the day. '
                f'For instance, the persona may say you are a girl that likes Legos and '
                f'today you are feeling sad. If you got the role of Karu, you should aim to be empathetic, interesting and knowledgeable.'
            )
        })
        agent_act = self.mturk_agent.act(timeout=self.opt['max_onboard_resp_time'])
        if self.check_timeout(agent_act):
            return


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

    def get_opponent(self, agent):
        return self.parlai_agent if agent == self.mturk_agent else self.mturk_agent

    def assign_conv_role(self):
        child_persona_text = gen_child_persona_sentence()
        robot = random.choice(robot_personas)
        robot_persona_text = (
            f'{robot["title"]} Karu. '
            f'{robot["description"]}'
        )
        if self.mturk_agent.id == 'CHILD':
            self.mturk_agent.persona_text = child_persona_text
            self.parlai_agent.persona_text = robot_persona_text
        else:
            self.mturk_agent.persona_text = robot_persona_text
            self.parlai_agent.persona_text = child_persona_text
        self.parlai_agent.observe({
            'text': 'your persona: ' + self.parlai_agent.persona_text,
            'persona': True
        })
        personaset_resp = self.parlai_agent.act()
        if personaset_resp:
            print(personaset_resp['text'], ' ', self.parlai_agent.persona_text)

    def parley(self):
        self.turn_index += 1
        acts = {}

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}
        """If at first turn, we need to give each agent some prior info if any like personas"""
        if self.turn_index == 1:
            control_msg['text'] = self.get_instruction(tag='start', agent=self.mturk_agent)
            control_msg['show_persona'] = True
            control_msg['persona_description'] = (
                    '<br>'
                    '<b><h3>Your Persona</h3></b>'
                    f"<ul><li>You're assigned with the following character: <br>"
                    f'<b><span style="color:blue">{self.mturk_agent.persona_text}</span></b></li>'
                    f'<li>You need to finish at least <b>' + str(self.n_turn) +
                    ' chat turns</b>, after that you can click the "Done" button '
                    'to end the chat.</li></ul>'
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
            if self.opt.get('bot_evaluation'):
                self.get_bot_evaluation(self.mturk_agent)
            control_msg['episode_done'] = True
            control_msg['text'] = self.get_instruction('end')
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
                    if self.opt.get('bot_evaluation'):
                        self.get_bot_evaluation(self.mturk_agent)
                    control_msg['text'] = self.get_instruction('end')
                    control_msg['episode_done'] = True
                    self.mturk_agent.observe(validate(control_msg))
                self.episodeDone = True
                return
            else:
                self.dialog.append({"turn_index": self.turn_index,
                                    "id": agent.id,
                                    "text": acts[agent.id]["text"]})
                for other_agent in [self.mturk_agent, self.parlai_agent]:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[agent.id]))

    def get_bot_evaluation(self, agent):
        agent.observe(validate({
            'id': 'SYSTEM',
            'text': (
                'We would like you to give some feedback on the conversation you had.'
            ),
            'episode_done': False
        }))
        questions = random.sample(ACUTE_EVAL_QUESTIONS, len(ACUTE_EVAL_QUESTIONS))
        self.bot_eval_by_worker = dict()
        self.bot_eval_by_worker[agent.id] = dict()
        for quest in questions:
            if quest["title"] == "Persona":
                eval_quest = quest["question"] + "<br><i>" + self.get_opponent(agent).persona_text + "</i>"
            else:
                eval_quest = quest["question"]
            option_list = random.sample(["Yes", "No"], 2)
            valid_answers = random.sample(list(range(1, 6)), 2)
            choice_list_html = ''.join(['<li>' + ch + '</li>' for ch in option_list])
            agent.observe(validate({
                'id': 'SYSTEM',
                'text': (
                    f'<b>{eval_quest}</b>'
                    '<br>'
                    f'<ul>{choice_list_html}</ul>'
                    '<br>'
                    f'Please send "<b>{valid_answers[0]} for {option_list[0]}</b>" or "<b>{valid_answers[1]} for {option_list[1]}.</b>"'
                ),
                'episode_done': False
            }))
            eval_act = agent.act(timeout=self.opt["max_resp_time"])
            if self.check_timeout(eval_act):
                return
            while eval_act["text"].strip().lower() not in [str(ans) for ans in valid_answers]:
                agent.observe(validate({
                    "id": "SYSTEM",
                    "text": f'Please read question carefully and send "<b>{valid_answers[0]} for {option_list[0]}</b>" or "<b>{valid_answers[1]} for {option_list[1]}.</b>"',
                    "episode_done": False,
                }))
                eval_act = agent.act(timeout=self.opt["max_resp_time"])
                if self.check_timeout(eval_act):
                    return
            act_index = int(eval_act["text"])
            self.bot_eval_by_worker[agent.id].update({
                quest["title"]: option_list[valid_answers.index(act_index)]
            })

    def get_instruction(self, tag, agent=None):
        if tag == 'start':
            return (
                '\nPlease chat with the other party. Your character is as follows::'
                f'\n<b><span style="color:blue">{agent.persona_text}</span></b>'
                '\nYou can also track the character description on the left.'
                '\n<b>Please try to match the length of other party\'s message. '
                'Share information relevant to character assigned and try to know other party as much as you can. '
                '</b>'
            )
        if tag == 'end':
            return 'Thanks for taking part in this HIT. If you like you can do more HITs.'
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
                'Your message is too short, please make it comparable to opponent message length.'
            )
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = (
                'Your message is too long, please make it comparable to opponent message length.'
            )
            ag.observe(validate(control_msg))
            return True
        return False

    def check_timeout(self, act):
        if (act['text'] in [TIMEOUT_MESSAGE, MTURK_DISCONNECT_MESSAGE, RETURN_MESSAGE]) and act['episode_done']:
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
        print("World shut down")

    def review_work(self):
        if self.opt.get('immediate_assignment_approve'):
            if hasattr(self.mturk_agent, 'not_approve'):
                pass
            else:
                self.mturk_agent.approve_work()

    def get_custom_task_data(self):
        return {'conversations': self.dialog,
                'worker_role': self.mturk_agent.id,
                'worker_persona': self.mturk_agent.persona_text,
                'bot_role': self.parlai_agent.id,
                'bot_persona': self.parlai_agent.persona_text,
                'bot_eval_by_worker': self.bot_eval_by_worker}
