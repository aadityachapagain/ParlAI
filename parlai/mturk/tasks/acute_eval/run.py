#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
import copy

from parlai.core.params import ParlaiParser
from parlai.mturk.core import mturk_utils, shared_utils
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once


DEFAULT_TASK_CONFIG = {
    'hit_title': 'Which Conversational Partner is Better?',
    'hit_description': 'Evaluate quality of conversations through comparison.',
    'hit_keywords': 'chat,evaluation,comparison,conversation',
}

AGENT_DISPLAY_NAME = 'RatingWorker'


def add_args(from_argv=False):
    """
    Add arguments to parser and either parse from commandline or initialize to defaults
    (for overriding in scripts)
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--annotations-per-pair',
        type=int,
        default=1,
        help='Number of annotations per conversation comparison pair',
    )
    argparser.add_argument(
        '--pairings-filepath',
        type=str,
        default=None,
        help='path to the file containing the task dictionaries',
    )
    argparser.add_argument(
        '--task-config',
        type=dict,
        default=DEFAULT_TASK_CONFIG,
        help='dict with keys "hit_title", "hit_description", "hit_keywords", '
        'determining how task is displayed on MTurk site',
    )
    argparser.add_argument(
        '--s1-choice',
        type=str,
        default='I would prefer to talk to <Speaker 1>',
        help='text next to speaker 1 radio button',
    )
    argparser.add_argument(
        '--s2-choice',
        type=str,
        default='I would prefer to talk to <Speaker 2>',
        help='text next to speaker 2 radio button',
    )
    argparser.add_argument(
        '--question',
        type=str,
        default='Who would you prefer to talk to for a long conversation?',
        help='question to present to turker for comparison (e.g. "Which speaker is better?")',
    )
    argparser.add_argument(
        '--block-on-onboarding-fail',
        type=bool,
        default=True,
        help='whether to block on onboarding failure',
    )
    argparser.add_argument(
        '--subtasks-per-hit',
        type=int,
        default=5,
        help='number of subtasks/comparisons to do per hit',
    )
    argparser.add_argument(
        '--onboarding-threshold',
        type=float,
        default=0.75,
        help='minimum accuracy on onboarding tasks, as a float 0-1.0',
    )
    argparser.add_argument('--seed', type=int, default=42, help='seed for random')
    argparser.add_argument(
        '--softblock-list-path',
        type=str,
        default=None,
        help='Path to list of workers to softblock, separated by line breaks',
    )
    argparser.set_defaults(allowed_conversation=1)
    if from_argv:
        return argparser.parse_args()
    else:
        return argparser.parse_args(args=[])


class AcuteEvaluator(object):
    """
    Run ACUTE Eval.

    Relevant args are parsed in the `setup_args` function above.
    """

    def __init__(self, opt):
        """
        Initialize the AcuteEvaluator.

        The following object attributes are used in running ACUTE Eval:

        ``onboarding_tasks``: A list of ALL available _onboarding_ comparison tasks

        ``desired_tasks``: A list of ALL available comparison tasks

        ``task_queue``: A queue of REMAINING tasks, from which HITs are constructed.

        ``worker_data``: A mapping from worker ID to data about the worker, including
        their tasks completed, conversations seen, and onboarding todo

        ``failed_onboard``:   The set of workers who have failed onboarding
        """
        random.seed(opt['seed'])
        self.opt = opt

        # add additional opt args
        self._supplement_opt()

        # class attributes
        self.onboarding_tasks: List[Dict] = []
        self.desired_tasks: List[Dict] = []
        self.task_queue: queue.Queue = queue.Queue()
        self.worker_data: Dict[str, Dict[str, List]] = {}
        self.failed_onboard: Set = set()

        # read in conversations data
        self._load_conversation_data()

        # setup the task queue
        self._setup_task_queue()

        # instantiate Manager
        self.manager = StaticMTurkManager(opt=self.opt)

    def _get_worker_data(self, worker_id: str) -> Dict[str, List]:
        """
        Return worker data if present, else a default dict.
        """
        onboarding_todo = list(range(len(self.onboarding_tasks)))
        random.shuffle(onboarding_todo)
        self.worker_data[worker_id] = self.worker_data.get(
            worker_id,
            {
                'tasks_completed': [],
                'conversations_seen': [],
                'onboarding_todo': onboarding_todo,
            },
        )
        return self.worker_data[worker_id]

    def _supplement_opt(self):
        """
        Add additional args to opt.

        Useful to add relevant options after args are parsed.
        """
        self.opt.update({
            'subtasks_per_hit': (len(self.opt.get('acute_questions'))
                                 if self.opt.get('acute_questions')
                                 else self.opt.get('subtasks_per_hit'))
        })

        self.opt.update(
            {
                'task': os.path.basename(os.path.dirname(os.path.abspath(__file__))),
                'task_description': {
                    'num_subtasks': self.opt['subtasks_per_hit'],
                    'question': self.opt['question'],
                },
                'frontend_version': 1,
            }
        )
        self.opt.update(self.opt['task_config'])

    def set_block_qual(self, task_group_id: str):
        """
        Set block qualification if necessary.

        :param task_group_id:
            task id used to set block qualification, if necessary.
        """
        if (
            self.opt['block_on_onboarding_fail']
            and self.opt['block_qualification'] is None
        ):
            self.opt['block_qualification'] = task_group_id
            warn_once(
                "No block_qualification set in opt, automatically creating "
                "new qualification {}".format(task_group_id)
            )

    def _load_conversation_data(self):
        """
        Load conversation data.

        Loads in the data from the pairs filepath.
        """
        pairs_path = self.opt.get('pairings_filepath')
        if not os.path.exists(pairs_path):
            raise RuntimeError('You MUST specify a valid pairings filepath')

        with open(pairs_path) as pf:
            for i, l in enumerate(pf.readlines()):
                convo_pair = json.loads(l.strip())
                eval_speakers = [
                    s
                    for d in convo_pair['dialogue_dicts']
                    for s in d['speakers']
                    if s in convo_pair['speakers_to_eval']
                ]
                # make sure order is preserved
                assert eval_speakers == convo_pair['speakers_to_eval']
                model_left_idx = random.choice([0, 1])
                task = {
                    'task_specs': {
                        's1_choice': self.opt['s1_choice'],
                        's2_choice': self.opt['s2_choice'],
                        'question': self.opt['question'],
                        'is_onboarding': convo_pair['is_onboarding'],
                        'model_left': {
                            'name': eval_speakers[model_left_idx],
                            'dialogue': convo_pair['dialogue_dicts'][model_left_idx][
                                'dialogue'
                            ],
                        },
                        'model_right': {
                            'name': eval_speakers[1 - model_left_idx],
                            'dialogue': convo_pair['dialogue_dicts'][
                                1 - model_left_idx
                            ]['dialogue'],
                        },
                    },
                    'pairing_dict': convo_pair,
                    'pair_id': i,
                }
                if convo_pair.get('is_onboarding'):
                    self.onboarding_tasks.append(task)
                else:
                    self.desired_tasks.append(task)

    def _setup_task_queue(self):
        """
        Fill task queue with conversation pairs.
        """
        for _i in range(self.opt['annotations_per_pair']):
            all_task_keys = list(range(len(self.desired_tasks)))
            random.shuffle(all_task_keys)
            for p_id in all_task_keys:
                self.task_queue.put(self.desired_tasks[p_id])

    def _get_dialogue_ids(self, task: Dict[str, Any]) -> List[int]:
        """
        Return the ids for the dialogues corresponding to a given task.

        :return dialogue_ids:
            A list of two ids which correspond to the id for each conversation
        """
        return task['pairing_dict']['dialogue_ids']

    def _poll_task_queue(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Poll task queue for tasks for a worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        num_attempts = 0
        while (not self.task_queue.empty()) and num_attempts < self.task_queue.qsize():
            try:
                next_task = self.task_queue.get()
            except queue.Empty:
                break
            num_attempts += 1

            pair_id = next_task['pair_id']
            dialogue_ids = self._get_dialogue_ids(next_task)

            # make sure worker has not seen these conversations before
            if pair_id not in worker_data['tasks_completed'] and all(
                d_id not in worker_data['conversations_seen'] for d_id in dialogue_ids
            ):
                # track tasks and conversations seen
                worker_data['tasks_completed'].append(pair_id)
                worker_data['conversations_seen'].extend(dialogue_ids)
                task_data.append(next_task)
                if len(task_data) == self.opt['subtasks_per_hit']:
                    return task_data
            else:
                self.task_queue.put(next_task)

        return task_data

    def _top_up_task_data(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Top up worker task data.

        This function is called if ``self.task_queue`` is exhausted but
        task_data for the worker is less than the `tasks_per_hit`.

        Make sure that all added tasks have not been seen by the worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        tasks_still_needed = self.opt['subtasks_per_hit'] - len(task_data)
        tasks_remaining = [
            t_id
            for t_id in range(len(self.desired_tasks))
            if t_id not in worker_data['tasks_completed']
        ]
        # get any pairings with conversations this worker has not seen to fill this hit
        additional_tasks = [
            t
            for t in tasks_remaining
            if all(
                d_id not in worker_data['conversations_seen']
                for d_id in self._get_dialogue_ids(self.desired_tasks[t])
            )
        ]
        if tasks_still_needed < len(additional_tasks):
            additional_tasks = random.sample(additional_tasks, tasks_still_needed)
        worker_data['tasks_completed'].extend(additional_tasks)

        for t in additional_tasks:
            worker_data['conversations_seen'].extend(
                self._get_dialogue_ids(self.desired_tasks[t])
            )
            task_data.append(self.desired_tasks[t])

        return task_data

    def attach_acute_questions(self, task_data):
        if len(task_data) <= len(self.opt['acute_questions']):
            questions = random.sample(list(self.opt['acute_questions']), len(task_data))
        else:
            questions = random.sample(list(self.opt['acute_questions']), len(self.opt['acute_questions']))
            questions.extend([random.choice(list(self.opt['acute_questions'])) for _ in
                              range(len(task_data) - len(self.opt['acute_questions']))])
        assert len(questions) == len(self.opt['acute_questions']), "Number of Acute Questions should match number of task data."
        for quest, task in zip(questions, task_data):
            task['task_specs'].update({
                's1_choice': self.opt['acute_questions'][quest]['s1_choice'],
                's2_choice': self.opt['acute_questions'][quest]['s2_choice'],
                'question': self.opt['acute_questions'][quest]['question'],
            })
        return task_data

    def get_new_task_data(self, worker_id: str) -> List[Dict[str, Any]]:
        """
        Get next task for worker.

        Returns the next onboarding task if worker hasn't finished them all,
        Otherwise finds a task from the queue they haven't seen

        If they've seen everything in the queue, spin up an
        extra task (one that was in the queue and is now saturated)

        :param worker_id:
            worker id

        :return task_data:
            A list of tasks for the worker to complete
        """
        tasks_per_hit = self.opt['subtasks_per_hit']
        # first add onboarding tasks
        task_data = self.get_onboarding_tasks(worker_id)

        # poll the task queue for more tasks
        if len(task_data) < tasks_per_hit:
            task_data = self._poll_task_queue(worker_id, task_data)

        # top up the task_data if we don't hit the desired tasks_per_hit
        if len(task_data) < tasks_per_hit:
            task_data = self._top_up_task_data(worker_id, task_data)

        if self.opt.get('acute_questions'):
            task_data = self.attach_acute_questions(task_data)
        return task_data

    def requeue_task_data(self, worker_id: str, task_data: List[Dict[str, Any]]):
        """
        Return task to task_queue.

        If the task is an onboarding task, indicate that the worker has
        another onboarding task to do.

        :param worker_id:
            worker id of worker who is returning task

        :param task_data:
            list of unfinished tasks to return to the queue.
        """
        worker_data = self._get_worker_data(worker_id)
        for subtask_data in task_data:
            if subtask_data['task_specs'].get('is_onboarding', False):
                worker_data['onboarding_todo'].append(subtask_data['pair_id'])
            else:
                self.task_queue.put(subtask_data)
                try:
                    worker_data['tasks_completed'].remove(subtask_data['pair_id'])
                    for d_id in self._get_dialogue_ids(subtask_data):
                        worker_data['conversations_seen'].remove(d_id)
                except ValueError:
                    # Task may have shown up in worker's task queue twice
                    # due to some unfortunate race condition
                    warn_once(f'could not remove task from worker {worker_id} history')

    def get_onboarding_tasks(self, worker_id: str) -> List[Dict[str, Any]]:
        """
        Get next onboarding task for given worker.

        :param worker_id:
            worker id

        :return:
            A list of onboarding tasks for the worker
        """
        if len(self.onboarding_tasks) == 0:
            return []

        worker_data = self._get_worker_data(worker_id)
        onboarding_todo = worker_data['onboarding_todo']
        if not onboarding_todo:
            # worker has completed all required onboarding tasks
            return []
        # get onboarding tasks for workers needing them
        num_tasks_to_return = min(len(onboarding_todo), self.opt['subtasks_per_hit'])
        onboarding_tasks_chosen = onboarding_todo[:num_tasks_to_return]
        worker_data['onboarding_todo'] = onboarding_todo[num_tasks_to_return:]
        return [self.onboarding_tasks[t_id] for t_id in onboarding_tasks_chosen]

    def check_and_update_worker_approval(
        self, worker_id: str, save_data: Dict[str, Any]
    ):
        """
        Soft block workers who fail onboarding tasks, keep track of their status.

        :param worker_id:
            worker id

        :param save_data:
            data from the worker's completed tasks
        """
        all_task_data = save_data['worker_data'][worker_id]['task_data']
        response_data = save_data['worker_data'][worker_id]['response']['task_data']
        num_onboarding_tasks = 0
        num_correct = 0

        for i in range(len(all_task_data)):
            is_onboarding = all_task_data[i]['pairing_dict'].get('is_onboarding', False)
            if not is_onboarding:
                # not an onboarding task, no need to check correctness
                continue
            worker_response = response_data[i]['speakerChoice']
            expected_response = all_task_data[i]['pairing_dict']['correct_answer']
            num_onboarding_tasks += 1
            if worker_response == expected_response:
                # count correct answers
                num_correct += 1
        if num_onboarding_tasks == 0:
            # no onboarding tasks found
            if worker_id in self.failed_onboard:
                # worker already failed onboarding, add pairings back to queue
                self.requeue_task_data(worker_id, all_task_data)
            return
        if (num_correct / num_onboarding_tasks) >= self.opt['onboarding_threshold']:
            # worker passed onboarding
            return
        # worker failed onboarding, soft block and record
        self.manager.soft_block_worker(worker_id)
        self.failed_onboard.add(worker_id)

    def softblock_workers(self):
        """
        Softblock workers if necessary.
        """
        if not self.opt['is_sandbox'] and self.opt['softblock_list_path'] is not None:
            softblock_list = set()
            with open(self.opt['softblock_list_path']) as f:
                for line in f:
                    softblock_list.add(line.strip())
            print(f'Will softblock {len(softblock_list):d} workers.')
            for w in softblock_list:
                try:
                    print('Soft Blocking {}\n'.format(w))
                    self.manager.soft_block_worker(w)
                except Exception as e:
                    print(f'Did not soft block worker {w}: {e}')
                time.sleep(0.1)

    def run(self):
        self.manager.setup_server(
            task_directory_path=os.path.dirname(os.path.abspath(__file__))
        )
        self.manager.set_onboard_function(onboard_function=None)
        task_group_id: str = None

        try:
            # Initialize run information
            self.manager.start_new_run()

            task_group_id = self.manager.task_group_id
            self.set_block_qual(task_group_id)
            self.manager.ready_to_accept_workers()
            self.manager.create_hits()

            def check_worker_eligibility(worker):
                return True

            def assign_worker_roles(workers):
                workers[0].id = AGENT_DISPLAY_NAME

            def run_conversation(mturk_manager, opt, workers):
                task_data = self.get_new_task_data(workers[0].worker_id)
                world = StaticMTurkTaskWorld(
                    opt, mturk_agent=workers[0], task_data=task_data
                )
                while not world.episode_done():
                    world.parley()

                world.shutdown()

                save_data = world.prep_save_data(workers)

                if not world.did_complete():
                    self.requeue_task_data(workers[0].worker_id, task_data)
                else:
                    if opt['block_on_onboarding_fail']:
                        # check whether workers failed onboarding
                        self.check_and_update_worker_approval(
                            workers[0].worker_id, save_data
                        )
                return save_data

            # Soft-block all chosen workers
            self.softblock_workers()
            print("This run id: {}".format(task_group_id))

            # Begin the task, allowing mturk_manager to start running the task
            # world on any workers who connect
            self.manager.start_task(
                eligibility_function=check_worker_eligibility,
                assign_role_function=assign_worker_roles,
                task_function=run_conversation,
            )
        finally:
            self.manager.expire_all_unassigned_hits()
            self.manager.shutdown()

        return task_group_id


class PersonaMatchingAcuteEvaluator(AcuteEvaluator):
    def __init__(self, opt, dedicated_workers=None):
        super(PersonaMatchingAcuteEvaluator, self).__init__(opt)
        self.mturk_page_url = None
        self.dedicated_workers = dedicated_workers
        self.dedicated_workers_qual_id = self._create_and_assign_dedicated_worker_qualification()
        self.task_queue: List = list(self.task_queue.queue)

    def _create_and_assign_dedicated_worker_qualification(self):
        mturk_utils.setup_aws_credentials()
        if not (self.opt.get('dedicated_worker_qualification') and self.dedicated_workers):
            return None
        qual_name = self.opt['dedicated_worker_qualification']
        qual_desc = (
            'Qualification given to filter workers to work on Conversation Comparision task.'
        )
        qual_id = mturk_utils.find_qualification(qual_name, self.opt['is_sandbox'])
        if qual_id:
            mturk_utils.delete_qualification(qual_id, self.opt['is_sandbox'])
        client = mturk_utils.get_mturk_client(self.opt['is_sandbox'])
        qual_id = client.create_qualification_type(
            Name=qual_name,
            Description=qual_desc,
            QualificationTypeStatus='Active',
        )['QualificationType']['QualificationTypeId']

        shared_utils.print_and_log(logging.INFO,
                                   f"Assigning {len(self.dedicated_workers)} workers dedicated worker qualification {qual_name}",
                                   should_print=True)
        for worker_id in self.dedicated_workers:
            mturk_utils.give_worker_qualification(worker_id, qual_id, is_sandbox=self.opt['is_sandbox'])

        return qual_id

    def _get_qualification_lists(self):
        if not self.dedicated_workers_qual_id:
            return None
        return [
            {
                'QualificationTypeId': self.dedicated_workers_qual_id,
                'Comparator': 'Exists',
                'ActionsGuarded': 'DiscoverPreviewAndAccept'
            }
        ]

    def _get_hit_notification_message(self):
        subject = "Which Conversational Partner is Better?"
        message = (
            "Thank you for your contribution in Child Companion Dialog HIT you did for us. "
            f"We have created another set of {self.opt['max_hits_per_worker']} HITs, only for you. "
            f"In this task, you'll read two conversation and compare two of the speakers based on "
            f"{len(self.opt.get('acute_questions')) if self.opt.get('acute_questions') else 'some'} questions "
            "we've put on the task. Please find the HITs using following link. "
            f"\nLink: {self.mturk_page_url} \n"
            "Note: We're experimenting with this tasks. If everything goes well we might come up with higher"
            " number of HITs with higher rewards."
        )

        return subject, message

    def _notify_dedicated_workers(self):
        if not (self.mturk_page_url and self.dedicated_workers):
            return
        client = mturk_utils.get_mturk_client(self.opt['is_sandbox'])
        worker_ids = [self.dedicated_workers[i: i + 100] for i in range(0, len(self.dedicated_workers), 100)]
        failures = []
        subject, message_text = self._get_hit_notification_message()
        shared_utils.print_and_log(logging.INFO,
                                   f"Sending emails to {len(self.dedicated_workers)} workers",
                                   should_print=True)
        for worker_ids_chunk in worker_ids:
            resp = client.notify_workers(
                Subject=subject, MessageText=message_text, WorkerIds=worker_ids_chunk
            )
            failures.extend(resp['NotifyWorkersFailureStatuses'])
        if failures:
            shared_utils.print_and_log(logging.WARN,
                                       f"Sending email to {len(failures)} workers failed...")
        return failures

    def _load_conversation_data(self):
        """
        Load conversation data.

        Loads in the data from the pairs filepath.
        """
        pairs_path = self.opt.get('pairings_filepath')
        if not os.path.exists(pairs_path):
            raise RuntimeError('You MUST specify a valid pairings filepath')

        with open(pairs_path) as pf:
            for i, l in enumerate(pf.readlines()):
                convo_pair = json.loads(l.strip())
                eval_speakers = [
                    s
                    for d in convo_pair['dialogue_dicts']
                    for s in d['speakers']
                    if s in convo_pair['speakers_to_eval']
                ]
                # make sure order is preserved
                assert eval_speakers == convo_pair['speakers_to_eval']
                model_left_idx = random.choice([0, 1])
                task = {
                    'task_specs': {
                        's1_choice': self.opt['s1_choice'],
                        's2_choice': self.opt['s2_choice'],
                        'question': self.opt['question'],
                        'is_onboarding': convo_pair['is_onboarding'],
                        'model_left': {
                            'name': eval_speakers[model_left_idx],
                            'dialogue': convo_pair['dialogue_dicts'][model_left_idx][
                                'dialogue'
                            ],
                        },
                        'persona': (convo_pair['dialogue_dicts'][model_left_idx][
                                       'context'][0]['text'][-1][14:]
                                    if convo_pair['dialogue_dicts'][model_left_idx].get('context')
                                    else ''),
                        'model_right': {
                            'name': eval_speakers[1 - model_left_idx],
                            'dialogue': convo_pair['dialogue_dicts'][
                                1 - model_left_idx
                            ]['dialogue'],
                        },
                    },
                    'pairing_dict': convo_pair,
                    'pair_id': i,
                }
                if convo_pair.get('is_onboarding'):
                    self.onboarding_tasks.append(task)
                else:
                    self.desired_tasks.append(task)

    def attach_acute_questions(self, task_data):
        question_keys = random.sample(list(self.opt['acute_questions']), len(self.opt['acute_questions']))
        task_data = [copy.deepcopy(task_data[0]) for _ in range(len(question_keys))]

        for question_key, task in zip(question_keys, task_data):
            if question_key.lower() == 'persona':
                question = self.opt['acute_questions'][question_key]['question']
            else:
                question = self.opt['acute_questions'][question_key]['question']

            task['task_specs'].update({
                's1_choice': self.opt['acute_questions'][question_key]['s1_choice'],
                's2_choice': self.opt['acute_questions'][question_key]['s2_choice'],
                'question': question,
                'question_type': question_key,
            })
            if question_key.lower() in ['contradictory', 'repeating'] and (
                    self.opt['acute_questions'][question_key].get('neutral_choice')):
                task['task_specs'].update({
                    'neutral_choice': self.opt['acute_questions'][question_key].get('neutral_choice')
                })
        return task_data

    def _supplement_opt(self):
        """
        Add additional args to opt.

        Useful to add relevant options after args are parsed.
        """
        self.opt.update({
            'subtasks_per_hit': (len(self.opt.get('acute_questions'))
                                 if self.opt.get('acute_questions')
                                 else self.opt.get('subtasks_per_hit'))
        })

        self.opt.update(
            {
                'task': os.path.basename(os.path.dirname(os.path.abspath(__file__))),
                'task_description': {
                    'num_subtasks': self.opt['subtasks_per_hit'],
                    'question': self.opt['question'],
                },
                'frontend_version': 1,
            }
        )
        self.opt.update(self.opt['task_config'])

    def _poll_task_queue(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Poll task queue for tasks for a worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        # num_attempts = 0
        # while (not self.task_queue.empty()) and num_attempts < self.task_queue.qsize():
        for next_task in self.task_queue:
            # try:
            #     next_task = self.task_queue.get()
            # except queue.Empty:
            #     break
            # num_attempts += 1

            pair_id = next_task['pair_id']
            dialogue_ids = self._get_dialogue_ids(next_task)

            # make sure worker has not seen these conversations before
            if pair_id not in worker_data['tasks_completed'] and all(
                d_id not in worker_data['conversations_seen'] for d_id in dialogue_ids
            ):
                # track tasks and conversations seen
                worker_data['tasks_completed'].append(pair_id)
                worker_data['conversations_seen'].extend(dialogue_ids)
                task_data.append(next_task)
                if len(task_data) == 1:
                    return task_data
            # else:
            #     self.task_queue.put(next_task)

        return task_data

    def _top_up_task_data(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Top up worker task data.

        This function is called if ``self.task_queue`` is exhausted but
        task_data for the worker is less than the `tasks_per_hit`.

        Make sure that all added tasks have not been seen by the worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        tasks_still_needed = 1 - len(task_data)
        tasks_remaining = [
            t_id
            for t_id in range(len(self.desired_tasks))
            if t_id not in worker_data['tasks_completed']
        ]
        # get any pairings with conversations this worker has not seen to fill this hit
        additional_tasks = [
            t
            for t in tasks_remaining
            if all(
                d_id not in worker_data['conversations_seen']
                for d_id in self._get_dialogue_ids(self.desired_tasks[t])
            )
        ]
        if tasks_still_needed < len(additional_tasks):
            additional_tasks = random.sample(additional_tasks, tasks_still_needed)
        worker_data['tasks_completed'].extend(additional_tasks)

        for t in additional_tasks:
            worker_data['conversations_seen'].extend(
                self._get_dialogue_ids(self.desired_tasks[t])
            )
            task_data.append(self.desired_tasks[t])

        return task_data

    def get_new_task_data(self, worker_id: str) -> List[Dict[str, Any]]:
        """
        Get next task for worker.

        Returns the next onboarding task if worker hasn't finished them all,
        Otherwise finds a task from the queue they haven't seen

        If they've seen everything in the queue, spin up an
        extra task (one that was in the queue and is now saturated)

        :param worker_id:
            worker id

        :return task_data:
            A list of tasks for the worker to complete
        """
        # tasks_per_hit = self.opt['subtasks_per_hit']
        # # first add onboarding tasks
        # task_data = self.get_onboarding_tasks(worker_id)
        #
        # # poll the task queue for more tasks
        # if len(task_data) < tasks_per_hit:
        task_data = self._poll_task_queue(worker_id, [])

        # top up the task_data if we don't hit the desired tasks_per_hit
        if len(task_data) < 1:
            task_data = self._top_up_task_data(worker_id, task_data)

        if self.opt.get('acute_questions'):
            task_data = self.attach_acute_questions(task_data)
        return task_data

    def run(self):
        self.manager.setup_server(
            task_directory_path=os.path.dirname(os.path.abspath(__file__))
        )
        self.manager.set_onboard_function(onboard_function=None)
        task_group_id: str = None

        try:
            # Initialize run information
            self.manager.start_new_run()

            task_group_id = self.manager.task_group_id
            self.set_block_qual(task_group_id)
            self.manager.ready_to_accept_workers()
            self.mturk_page_url = self.manager.create_hits(qualifications=self._get_qualification_lists())
            if self.dedicated_workers_qual_id and self.dedicated_workers:
                self._notify_dedicated_workers()

            def check_worker_eligibility(worker):
                return True

            def assign_worker_roles(workers):
                workers[0].id = AGENT_DISPLAY_NAME

            def run_conversation(mturk_manager, opt, workers):
                task_data = self.get_new_task_data(workers[0].worker_id)
                world = StaticMTurkTaskWorld(
                    opt, mturk_agent=workers[0], task_data=task_data, act_timeout=True
                )
                while not world.episode_done():
                    world.parley()

                world.shutdown()

                save_data = world.prep_save_data(workers)

                if not world.did_complete():
                    # self.requeue_task_data(workers[0].worker_id, task_data)
                    pass
                else:
                    if opt['block_on_onboarding_fail']:
                        # check whether workers failed onboarding
                        self.check_and_update_worker_approval(
                            workers[0].worker_id, save_data
                        )
                return save_data

            # Soft-block all chosen workers
            self.softblock_workers()
            print("This run id: {}".format(task_group_id))

            # Begin the task, allowing mturk_manager to start running the task
            # world on any workers who connect
            self.manager.start_task(
                eligibility_function=check_worker_eligibility,
                assign_role_function=assign_worker_roles,
                task_function=run_conversation,
            )
        finally:
            self.manager.expire_all_unassigned_hits()
            self.manager.shutdown()

        return task_group_id

    def _setup_task_queue(self):
        all_task_keys = list(range(len(self.desired_tasks)))
        random.shuffle(all_task_keys)
        for p_id in all_task_keys:
            for _i in range(self.opt['annotations_per_pair']):
                self.task_queue.put(self.desired_tasks[p_id])


if __name__ == '__main__':
    args = add_args(from_argv=True)
    runner = AcuteEvaluator(args)
    runner.run()
