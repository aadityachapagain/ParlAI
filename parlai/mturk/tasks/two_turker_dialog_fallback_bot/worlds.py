import random
import logging

import numpy as np
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkTaskWorld, MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE, RETURN_MESSAGE, MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core import mturk_utils
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.one_sided_acute_eval_questions import ACUTE_EVAL_QUESTIONS
import parlai.mturk.core.shared_utils as shared_utils
from .context import Context


class QualificationTestOnboardWorld(MTurkOnboardWorld):
    def __init__(self, opt, mturk_agent, pass_qual_id, fail_qual_id):
        super(QualificationTestOnboardWorld, self).__init__(opt, mturk_agent)
        self.opt = opt
        self.pass_qual_id = pass_qual_id
        self.fail_qual_id = fail_qual_id
        self.first_time = not mturk_utils.check_worker_qualification_exist(
            self.pass_qual_id,
            self.mturk_agent.worker_id,
            is_sandbox=self.opt['is_sandbox']
        )
        self.pass_qual_test = None if self.first_time else True  # None==>neither pass nor fail(first time in this HIT)
        self.mturk_agent.context = Context().gen_context()
        self.mturk_agent.role = random.choice(['CHILD', 'KARU'])

    def parley(self):
        if self.first_time:
            # Check qualification result
            if self.send_qualification_question():
                shared_utils.print_and_log(logging.INFO,
                                           f"Worker {self.mturk_agent.worker_id} passed the qualification test.",
                                           should_print=True)
                self.pass_qual_test = True
                self.assign_qualification(self.pass_qual_id)
                self.send_task_instruction()
            else:
                shared_utils.print_and_log(logging.INFO,
                                           f"Worker {self.mturk_agent.worker_id} failed the qualification test.",
                                           should_print=True)
                self.pass_qual_test = False
                self.assign_qualification(self.fail_qual_id)
                # if Failed expire hit
                self.expire_hit((
                    'Sorry you\'ve failed our qualification test task. You cannot proceed to main task and other subsequent HITs,\n'
                    'This HIT is now expired. Please return this HIT.'
                ))
        else:
            # Worker is good to go to main task
            self.send_task_instruction()
        self.episodeDone = True

    def send_qualification_question(self):
        choice_answer = random.sample(list(self.mturk_agent.context['conv_theme']['qual_test_choices'].keys()), 3)
        correct_answer_index = choice_answer.index('correct_option') + 1
        choice_list_html = ''.join(
            ['<li>' + self.mturk_agent.context['conv_theme']['qual_test_choices'][op] + '</li>' for op in
             choice_answer])
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'text': (
                'You\'re here for the first time. Please read the description and answer following qualification question.'
                '\n'
                'In the chat, we want you to be as <b>specific</b> and helpful to the child as you can. \n'
                f'"<b>The other party wants to talk about {self.mturk_agent.context["conv_theme"]["theme"]}"</b>\n'
                f'Which is the <b>best</b> way to start the conversation?\n'
                f'<ol>{choice_list_html}</ol>'
                '<br>'
                f'Only numbers from <b>{1}</b> to <b>{len(choice_answer)}</b> are valid'
            ),
            'onboard_message': (
                '<br>'
                '<b><h4>Qualification Test</h4></b>'
                '<br>'
                'You\'re here for the first time. We need you to complete a '
                'qualification test before you proceed to the main task. \n'
                'Read the qualification test instruction carefully. \n'
                'Careful once you submit the answer you cannot change the answer.'
            )
        })
        agent_act = self.mturk_agent.act(timeout=300)
        if self.check_errors(agent_act):
            return False
        while agent_act['text'].strip() not in [str(i) for i in range(1, len(choice_answer) + 1)]:
            self.mturk_agent.observe({
                'id': 'SYSTEM',
                'text': f'Only numbers from {1} to {len(choice_answer)} are valid.\nResubmit your answer.'
            })
            agent_act = self.mturk_agent.act(timeout=300)
            if self.check_errors(agent_act):
                return False

        if agent_act['text'] == str(correct_answer_index):
            return True
        else:
            return False

    def send_task_instruction(self):
        first_time_message = (
            'Congratulations you completed qualification task.\n'
            '<b>Now, Please read the instructions on left carefully and '
            'keep in mind your persona and conversation theme. When you are ready '
            'send anything to continue.</b>'
        )
        old_turker_message = (
            'Welcome back! You\'ve already completed our qualification task. \n'
            '<b>Please read the instructions on left carefully and '
            'keep in mind your persona and conversation theme. When you are ready '
            'send anything to continue.</b>'
        )
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'qual_test_pass': True,
            'text': first_time_message if self.first_time else old_turker_message,
            'onboard_message': (
                '<b><h3>Task Instruction</h3></b>'
                f"<ul>"
                f'<li><b><span style="color:red">In this conversation, {self.mturk_agent.context["conv_theme"]["theme_sentence"]}</li>'
                f"<li>You're assigned with the following character: <br>"
                f'<ul><li><b><span style="color:blue">{self.mturk_agent.context["personas"]["child_persona" if self.mturk_agent.role == "CHILD" else "robot_persona"]}</span></b></li></ul>'
                '<li>Stick to the above character and topic of the conversation.</li>'
                f'</ul>'
                '<br>'
            )
        })
        agent_act = self.mturk_agent.act(timeout=self.opt['max_onboard_resp_time'])
        if self.check_errors(agent_act):
            return

    def check_errors(self, act):
        if act['text'] in [TIMEOUT_MESSAGE, RETURN_MESSAGE, MTURK_DISCONNECT_MESSAGE] and act['episode_done']:
            self.episodeDone = True
            return True
        else:
            return False

    def assign_qualification(self, qaul_id):
        mturk_utils.give_worker_qualification(
            self.mturk_agent.worker_id,
            qaul_id,
            is_sandbox=self.opt['is_sandbox']
        )
        shared_utils.print_and_log(logging.INFO,
                                   f"Assigning worker {self.mturk_agent.worker_id} {'pass' if qaul_id == self.pass_qual_id else 'fail'} qualification",
                                   should_print=True)

    def expire_hit(self, message=None):
        self.mturk_agent.mturk_manager.force_expire_hit(self.mturk_agent.worker_id,
                                                        self.mturk_agent.assignment_id,
                                                        text=message)


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
        child_persona_text = self.mturk_agent.context['personas']['child_persona']
        robot_persona_text = self.mturk_agent.context['personas']['robot_persona']
        self.mturk_agent.theme_sentence = self.mturk_agent.context['conv_theme']['theme_sentence']
        if self.mturk_agent.id == 'CHILD':
            self.mturk_agent.persona_text = child_persona_text
            self.parlai_agent.persona_text = robot_persona_text
        else:
            self.mturk_agent.persona_text = robot_persona_text
            self.parlai_agent.persona_text = child_persona_text

        bot_persona = 'your persona: ' + self.mturk_agent.context['conv_theme']['theme_sentence'] \
                      + ' \\nyour persona: ' + self.parlai_agent.persona_text
        self.parlai_agent.observe({
            'text': bot_persona,
            'persona': True
        })
        personaset_resp = self.parlai_agent.act()
        if personaset_resp:
            print(personaset_resp['text'], ' ', bot_persona)

    def parley(self):
        self.turn_index += 1
        acts = {}

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}
        """If at first turn, we need to give each agent some prior info if any like personas"""
        if self.turn_index == 1:
            control_msg['text'] = self.get_instruction(tag='start', agent=self.mturk_agent)
            control_msg['show_persona'] = True
            control_msg['persona_description'] = (
                '<b><h3>Task Instruction</h3></b>'
                f"<ul>"
                f'<li><b><span style="color:red">In this conversation, {self.mturk_agent.context["conv_theme"]["theme_sentence"]}</li>'
                f"<li>You're assigned with the following character: <br>"
                f'<ul><li><b><span style="color:blue">{self.mturk_agent.context["personas"]["child_persona" if self.mturk_agent.id == "CHILD" else "robot_persona"]}</span></b></li></ul>'
                '<li>Stick to the above character and topic of the conversation.</li>'
                f'</ul>'
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
                
                if hasattr(self.parlai_agent, 'check_offensive'):
                    while self.parlai_agent.check_offensive(acts[agent.id]['text']):
                        agent.observe({
                            'id': 'SYSTEM',
                            'episode_done': False,
                            'text': (
                                'Your message was detected offensive. Please follow the instruction '
                                'and send non-offensive message.'
                            )
                        })
                        acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'])
                        if self.check_timeout(acts[agent.id]):
                            return

            else:
                acts[agent.id] = agent.act()

            if self.turn_index > 1 and agent == self.mturk_agent:
                # only check if message is too short/long except on first message
                while self.is_msg_tooshortlong(acts[agent.id], agent):
                    acts[agent.id] = agent.act(timeout=self.opt['max_resp_time'])
                    if hasattr(self.parlai_agent, 'check_offensive'):
                        while self.parlai_agent.check_offensive(acts[agent.id]['text']):
                            agent.observe({
                                'id': 'SYSTEM',
                                'episode_done': False,
                                'text': (
                                    'Your message was detected offensive. Please follow the instruction '
                                    'and send non-offensive message.'
                                )
                            })
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
                '\nPlease chat with the other party. '
                f'\n<b><span style="color:red">In this conversation, {agent.theme_sentence}</span></b>'
                '\nYour character is as follows:'
                f'\n<b><span style="color:blue">{agent.persona_text}</span></b>'
                '\nYou can also track the character description on the left.'
                '\n<b>Please try to match the length of other party\'s message. '
                'Share information relevant to character assigned and try to know other party as much as you can while adhering to persona and conversation theme. '
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
                'Your message is too short, please make it comparable to other party\'s message length.'
            )
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = (
                'Your message is too long, please make it comparable to other party\'s message length.'
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
                'bot_role': self.parlai_agent.id,
                'context': self.mturk_agent.context,
                'bot_eval_by_worker': self.bot_eval_by_worker}
