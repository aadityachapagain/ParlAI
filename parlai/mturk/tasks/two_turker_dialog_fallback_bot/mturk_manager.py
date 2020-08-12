import logging
import time
import threading

from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.shared_utils as shared_utils
from parlai.mturk.core.mturk_manager import MTurkManager, WORLD_START_TIMEOUT
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler


class MturkManagerWithWaitingPoolTimeout(MTurkManager):
    def start_task(self, eligibility_function, assign_role_function, task_function):
        """
        Handle running a task by checking to see when enough agents are in the pool to
        start an instance of the task.

        Continue doing this until the desired number of conversations is had.
        """
        assert self.task_state >= self.STATE_HITS_MADE, (
            'Must have launched HITs with `mturk_manager.create_hits`'
            ' to start the task'
        )
        if callable(eligibility_function):
            # Convert legacy eligibility_functions to the new format
            eligibility_function = {'multiple': False, 'func': eligibility_function}
        else:
            # Ensure the eligibility function is valid
            if 'func' not in eligibility_function:
                shared_utils.print_and_log(
                    logging.CRITICAL, "eligibility_function has no 'func'. Cancelling."
                )
                raise Exception(
                    'eligibility_function dict must contain a `func` field '
                    'containing the actual function.'
                )
            elif not callable(eligibility_function['func']):
                shared_utils.print_and_log(
                    logging.CRITICAL,
                    "eligibility_function['func'] not a function. Cancelling.",
                )
                raise Exception(
                    "eligibility_function['func'] must contain a function. "
                    "If eligibility_function['multiple'] is set, it should "
                    "filter through the list of workers and only return those "
                    "that are currently eligible to participate. If it is not "
                    "set, it should take in a single worker and return whether"
                    " or not they are eligible."
                )
            if 'multiple' not in eligibility_function:
                eligibility_function['multiple'] = False

        def _task_function(opt, agents, conversation_id):
            """
            Wait for agents to join the world, then run task function.
            """
            shared_utils.print_and_log(
                logging.INFO, 'Starting task {}...'.format(conversation_id)
            )
            shared_utils.print_and_log(
                logging.DEBUG, 'Waiting for all agents to join the conversation...'
            )
            start_time = time.time()
            while True:
                all_joined = True
                for agent in agents:
                    # check the status of an individual agent assignment
                    if agent.get_status() != AssignState.STATUS_IN_TASK:
                        all_joined = False
                if all_joined:
                    break
                if time.time() - start_time > WORLD_START_TIMEOUT:
                    # We waited but not all agents rejoined, throw agents
                    # back into the waiting pool. Stragglers will disconnect
                    # from there
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Timeout waiting for {}, move back to waiting'.format(
                            conversation_id
                        ),
                    )
                    self._move_agents_to_waiting(agents)
                    return
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

            shared_utils.print_and_log(
                logging.INFO,
                'All agents joined the conversation {}!'.format(conversation_id),
            )
            self.started_conversations += 1
            save_data = task_function(mturk_manager=self, opt=opt, workers=agents)

            if save_data is not None:
                MTurkDataHandler.save_world_data(
                    save_data,
                    self.task_group_id,
                    conversation_id,
                    sandbox=self.is_sandbox,
                )

            # Delete extra state data that is now unneeded
            for agent in agents:
                agent.clear_messages()

            # Count if it's a completed conversation
            if self._no_agents_incomplete(agents):
                self.completed_conversations += 1
            if self.opt['max_connections'] > 0:  # If using a conv cap
                if self.accepting_workers:  # if still looking for new agents
                    for agent in agents:
                        if agent.submitted_hit():
                            self.create_additional_hits(1)

        if self.db_logger is not None:
            self._maintain_hit_status()
        while not self.is_shutdown:
            if self.has_time_limit:
                self._check_time_limit()
            # Loop forever starting task worlds until desired convos are had
            with self.agent_pool_change_condition:
                valid_agents = self._get_unique_pool(eligibility_function)
                needed_agents = len(self.mturk_agent_ids)
                if len(valid_agents) >= needed_agents:
                    # enough agents in pool to start new conversation
                    self.conversation_index += 1
                    new_conversation_id = 't_{}'.format(self.conversation_index)

                    # Add the required number of valid agents to the conv
                    agents = [a for a in valid_agents[:needed_agents]]
                    assign_role_function(agents)
                    # Allow task creator to filter out agents and run
                    # versions of the task that require fewer agents
                    agents = [a for a in agents if a.id is not None]
                    for agent in agents:
                        self.worker_manager.change_agent_conversation(
                            agent=agent,
                            conversation_id=new_conversation_id,
                            new_agent_id=agent.id,
                        )
                        # Remove selected agents from the pool
                        self._remove_from_agent_pool(agent)

                    # Start a new thread for this task world
                    task_thread = threading.Thread(
                        target=_task_function,
                        args=(self.opt, agents, new_conversation_id),
                        name='task-{}'.format(new_conversation_id),
                    )
                    task_thread.daemon = True
                    task_thread.start()
                    self.task_threads.append(task_thread)
                elif self.opt.get('waiting_pool_time'):
                    for ag in valid_agents:
                        if (hasattr(ag, "onboard_leave_time") and (
                                time.time() - ag.onboard_leave_time > self.opt['waiting_pool_time'])):
                            # Than make conversation with bot
                            self.conversation_index += 1
                            new_conversation_id = 't_bot_{}'.format(self.conversation_index)

                            # Allow task creator to filter out agents and run
                            # versions of the task that require fewer agents
                            assign_role_function([ag])
                            if ag.id is not None:
                                self.worker_manager.change_agent_conversation(
                                    agent=ag,
                                    conversation_id=new_conversation_id,
                                    new_agent_id=ag.id,
                                )
                                self._remove_from_agent_pool(ag)
                                task_thread = threading.Thread(
                                    target=_task_function,
                                    args=(self.opt, [ag], new_conversation_id),
                                    name='task-bot-{}'.format(new_conversation_id),
                                )
                                task_thread.daemon = True
                                task_thread.start()
                                self.task_threads.append(task_thread)

            # Once we've had enough conversations, finish and break
            compare_count = self.started_conversations
            if self.opt['count_complete']:
                compare_count = self.completed_conversations
            if compare_count >= self.num_conversations:
                self.accepting_workers = False
                shared_utils.print_and_log(logging.INFO,
                                           f"Desired number of conversations {self.num_conversations} completed.....",
                                           should_print=True)
                self.expire_all_unassigned_hits()
                self._expire_onboarding_pool()
                self._expire_agent_pool()
                # Wait for all conversations to finish, then break from
                # the while loop
                shared_utils.print_and_log(logging.INFO, "Waiting for all conversations to finish.....",
                                           should_print=True)
                for thread in self.task_threads:
                    thread.join(timeout=self.opt['assignment_duration_in_seconds'])
                alive_threads = [thread for thread in self.task_threads if thread.isAlive()]
                shared_utils.print_and_log(logging.INFO, f"Continuing with {len(alive_threads)} alive threads....",
                                           should_print=True)
                break
            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)
