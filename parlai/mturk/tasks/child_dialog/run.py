import faulthandler; faulthandler.enable()

import os
import time
import yaml
import random
import logging
from threading import Thread
from tqdm import tqdm

from parlai.core.params import ParlaiParser
from parlai.mturk.core import mturk_utils
from parlai.core.agents import create_agent
from parlai.mturk.tasks.child_dialog.worlds import (
    QualificationTestOnboardWorld,
    InteractParlAIModelWorld
)
from parlai.mturk.tasks.child_dialog.mturk_manager import MturkManagerWithWaitingPoolTimeout
from parlai.mturk.tasks.child_dialog.task_config import task_config
import parlai.mturk.core.shared_utils as shared_utils
from parlai.mturk.tasks.child_dialog.agents import BotAgent
from parlai.mturk.tasks.child_dialog.dedicated_workers import ReviewGSheet
from parlai.utils.safety import OffensiveLanguageClassifier


def setup_args():
    argparser = ParlaiParser(False, True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    # argparser.add_argument(
    #     '--bot-host',
    #     dest='bot_host',
    #     required=True,
    #     help='IP Address of the bot API'
    # )
    # argparser.add_argument(
    #     '--bot-port',
    #     dest='bot_port',
    #     default=8000,
    #     help='Port address of Bot agent'
    # )
    # argparser.add_argument(
    #     '--bot-username',
    #     dest='bot_username',
    #     required=True,
    #     help='username to use for authentication in bot API'
    # )
    # argparser.add_argument(
    #     '--bot-password',
    #     dest='bot_password',
    #     required=True,
    #     help='password to use for authentication in bot API'
    # )
    argparser.add_argument(
        '--gsheet-credentials',
        dest='ghseet_credentials',
        required=True,
        help='path to gsheet credentials json file'
    )
    argparser.add_argument(
        '--safety',
        type=str,
        default='string_matcher',
        choices={'none', 'string_matcher', 'classifier', 'all'},
        help='Apply safety filtering to messages',
    )
    argparser.add_argument(
        '--custom-data-dir',
        dest='custom_data_dir',
        default=None,
        type=str,
        help='Mturk data path from where data is to be shown in dashboard.'
    )
    parsed_args = argparser.parse_args()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_args['task'] = os.path.basename(task_dir)
    parsed_args.update(task_config)

    with open(os.path.join(task_dir, 'config.yml')) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    parsed_args.update(cfg)

    parsed_args_str = '\n'.join([str(k) + ': ' + str(v) for k, v in parsed_args.items()])
    shared_utils.print_and_log(logging.INFO,
                               f"Parameters for the run.....\n {parsed_args_str}",
                               should_print=True)

    return parsed_args


def create_passfail_qualification(opt):
    mturk_utils.setup_aws_credentials()
    qual_pass_name = f'{opt["qual_test_qualification"]}Pass'
    qual_pass_desc = (
        'Qualification for a worker correctly completing the '
        'child companion dialog qualification test task.'
    )
    pass_qual_id = mturk_utils.find_or_create_qualification(
        qual_pass_name, qual_pass_desc, opt['is_sandbox']
    )

    qual_fail_name = f'{opt["qual_test_qualification"]}Fail'
    qual_fail_desc = (
        'Qualification for a worker not correctly completing the '
        'child companion dialog qualification test task.'
    )
    fail_qual_id = mturk_utils.find_or_create_qualification(
        qual_fail_name, qual_fail_desc, opt['is_sandbox']
    )
    shared_utils.print_and_log(logging.INFO,
                               f"Created Pass Qualification {pass_qual_id} and fail qualification {fail_qual_id}",
                               should_print=True)
    return pass_qual_id, fail_qual_id


def create_and_assign_dedicated_worker_qualification(opt, dedicated_workers):
    qual_name = f'{opt["dedicated_worker_qualification"]}'
    qual_desc = (
        'Qualification given to golden workers performing very well in child companion dialog. '
        'This qualification will be used to allocate HIT to these golden workers.'
    )
    qual_id = mturk_utils.find_qualification(qual_name, opt['is_sandbox'])
    if qual_id:
        mturk_utils.delete_qualification(qual_id, opt['is_sandbox'])
        shared_utils.print_and_log(logging.INFO,
                                   f"Deleted previous dedicated worker qualification {opt['dedicated_worker_qualification']}({qual_id})",
                                   should_print=True)
    client = mturk_utils.get_mturk_client(opt['is_sandbox'])
    qual_id = client.create_qualification_type(
        Name=qual_name,
        Description=qual_desc,
        QualificationTypeStatus='Active',
    )['QualificationType']['QualificationTypeId']
    shared_utils.print_and_log(logging.INFO,
                               f"Created dedicated worker qualification named {qual_name} with id {qual_id}",
                               should_print=True)
    shared_utils.print_and_log(logging.INFO,
                               f"Assigning {len(dedicated_workers)} workers the qualification {qual_id}",
                               should_print=True)
    for worker_id in tqdm(dedicated_workers):
        mturk_utils.give_worker_qualification(worker_id, qual_id, is_sandbox=opt['is_sandbox'])

    return qual_id


def prepare_dedicated_workers(pass_qual_id, dedicated_workers, is_sandbox):
    """Assign pass qualification if dedicated workers are from non qualification runs"""
    if len(dedicated_workers) > 20:
        shared_utils.print_and_log(logging.INFO,
                                   "Received more than 25 workers in batch. Selected 25 random workers.......")
        dedicated_workers = random.sample(dedicated_workers, 20)

    qual_pass_workers = mturk_utils.list_workers_with_qualification_type(pass_qual_id, is_sandbox)
    for dedicated_worker in dedicated_workers:
        if dedicated_worker not in qual_pass_workers:
            mturk_utils.give_worker_qualification(dedicated_worker, pass_qual_id, is_sandbox=is_sandbox)

    try:
        client = mturk_utils.get_mturk_client(is_sandbox)
        workers_assignments = mturk_utils.list_workers_assignments(dedicated_workers, is_sandbox, client=client)
        assignments_list = []
        for worker_id, assignments in workers_assignments.items():
            assignments_list.extend(assignments)
        mturk_utils.approve_list_of_assignments(assignments_list, is_sandbox, client=client)
    except Exception as e:
        shared_utils.print_and_log(logging.WARN, f"Approving dedicated workers assignments got error: {repr(e)}",
                                   should_print=True)
        shared_utils.print_and_log(logging.WARN, "Continuing without approving dedicated workers assignment...........",
                                   should_print=True)

    return dedicated_workers


def email_workers(worker_ids, subject, message_text, is_sandbox):
    if not isinstance(worker_ids, list):
        worker_ids = [worker_ids]
    client = mturk_utils.get_mturk_client(is_sandbox)
    worker_ids = [worker_ids[i: i + 100] for i in range(0, len(worker_ids), 100)]
    failures = []
    shared_utils.print_and_log(logging.INFO,
                               "Sending email to workers.........")
    for worker_ids_chunk in tqdm(worker_ids):
        resp = client.notify_workers(
            Subject=subject, MessageText=message_text, WorkerIds=worker_ids_chunk
        )
        failures.extend(resp['NotifyWorkersFailureStatuses'])

    if failures:
        shared_utils.print_and_log(logging.WARN,
                                   f"Sending email to {len(failures)} workers failed...")
    return failures


def get_hit_notification_message(hit_link):
    subject = "Chat as a Child: High reward HITs exclusive to you"
    message = (
        "Thank you for your outstanding performance in the previous HIT you did for us."
        "You followed the theme of the conversation, were specific and took initiative."        "So we have created a set of 20 HITs, only for you."
        "Please find the HITs using following link."
        f"\nLink: {hit_link} \n"
        "Note: If you can't find the HIT please wait for few moments and retry. "
        "Also, above link is valid for 24 hours only."
    )
    return subject, message


def send_hit_notification(worker_ids, hit_link, is_sandbox):
    subject, message = get_hit_notification_message(hit_link)
    _ = email_workers(worker_ids, subject, message, is_sandbox)


def single_run(opt,
               bot_agent,
               offensive_language_classifier,
               pass_qual_id,
               fail_qual_id,
               dedicated_worker_qual_id=None,
               dedicated_worker_ids=None):
    mturk_agent_ids = ['CHILD', 'KARU']
    mturk_manager = MturkManagerWithWaitingPoolTimeout(opt=opt,
                                                       mturk_agent_ids=[random.choice(mturk_agent_ids)],
                                                       use_db=True)
    mturk_manager.setup_server()

    def run_onboard(worker):
        world = QualificationTestOnboardWorld(opt=opt,
                                              mturk_agent=worker,
                                              pass_qual_id=pass_qual_id,
                                              fail_qual_id=fail_qual_id)
        while not world.episode_done():
            world.parley()
        world.shutdown()
        worker.onboard_leave_time = time.time()
        return world.prep_save_data([worker])

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        mturk_manager.start_new_run()
        mturk_manager.ready_to_accept_workers()

        agent_qualifications = [
            {
                'QualificationTypeId': fail_qual_id,
                'Comparator': 'DoesNotExist',
                'ActionsGuarded': 'DiscoverPreviewAndAccept',
            },
        ]
        if dedicated_worker_qual_id:
            agent_qualifications.append(
                {  # Dedicated worker qualification
                    'QualificationTypeId': dedicated_worker_qual_id,
                    'Comparator': 'Exists',
                    'ActionsGuarded': 'DiscoverPreviewAndAccept'
                }
            )
        else:
            agent_qualifications.append(
                {  # Min HIT approval rate
                    'QualificationTypeId': '000000000000000000L0',
                    'Comparator': 'GreaterThanOrEqualTo',
                    'IntegerValues': [opt['min_hit_approval_rate']],
                    'ActionsGuarded': 'DiscoverPreviewAndAccept'
                }
            )

        if not opt['is_sandbox']:
            agent_qualifications.extend([
                {
                    'QualificationTypeId': '00000000000000000071',
                    'Comparator': 'In',
                    'LocaleValues': [{'Country': country} for country in opt['allowed_countries']],
                    'ActionsGuarded': 'DiscoverPreviewAndAccept',
                },
            ])
        mturk_page_url = mturk_manager.create_hits(qualifications=agent_qualifications)

        if dedicated_worker_ids:
            send_hit_notification(dedicated_worker_ids, mturk_page_url, opt['is_sandbox'])

        def check_workers_eligibility(workers):
            return workers

        eligibility_function = {'func': check_workers_eligibility, 'multiple': True}

        def assign_worker_roles(workers):
            for worker in workers:
                worker.id = worker.role

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            shared_utils.print_and_log(logging.INFO, f"Launching conversation for {workers[0].worker_id}")
            if workers[0].id == 'CHILD':
                bot_agent_id = 'KARU'
            else:
                bot_agent_id = 'CHILD'
            bot = BotAgent(opt, bot_agent, bot_agent_id, mturk_manager.task_group_id,
                           offensive_language_classifier=offensive_language_classifier)
            world = InteractParlAIModelWorld(opt, workers[0], bot)

            while not world.episode_done():
                world.parley()

            world.shutdown()
            world.review_work()

            return world.prep_save_data(workers)

        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except BaseException as e:
        shared_utils.print_and_log(logging.WARN, f"Continuing with error: {repr(e)}", should_print=True)
        raise
    finally:
        if opt.get("delete_qual_test_qualification"):
            mturk_utils.delete_qualification(pass_qual_id, opt['is_sandbox'])
            mturk_utils.delete_qualification(fail_qual_id, opt['is_sandbox'])
            shared_utils.print_and_log(logging.INFO, "Deleted qualification..............", should_print=True)

        return mturk_manager


def run_final_job(manager):
    shared_utils.print_and_log(logging.INFO, f"Running Final Job of {manager.task_group_id}", should_print=True)
    manager.shutdown()


def main(opt):
    # Qualifications
    mturk_utils.setup_aws_credentials()
    pass_qual_id, fail_qual_id = create_passfail_qualification(opt)
    if opt.get("dedicated_worker_run"):
        if opt.get("max_hits_limit_in_a_run_only"):
            max_submission_qual_id = mturk_utils.find_qualification(opt['unique_qual_name'], opt['is_sandbox'])
            if max_submission_qual_id:
                mturk_utils.delete_qualification(max_submission_qual_id, opt['is_sandbox'])
                shared_utils.print_and_log(logging.INFO,
                                           f"Deleted max submissions qualification {opt['unique_qual_name']}",
                                           should_print=True)

        dedicated_workers_list = ReviewGSheet(opt['ghseet_credentials']).get_golden_workers_list()
        dedicated_workers_list = prepare_dedicated_workers(pass_qual_id, dedicated_workers_list, opt['is_sandbox'])
        dedicated_worker_qual_id = create_and_assign_dedicated_worker_qualification(opt, dedicated_workers_list)

        opt['num_conversations'] = opt['max_hits_per_worker'] * len(dedicated_workers_list)
    else:
        dedicated_workers_list = None
        dedicated_worker_qual_id = None

    bot_agent = create_agent(opt, requireModelExists=True)
    if opt['safety'] == 'classifier' or opt['safety'] == 'all':
        offensive_language_classifier = OffensiveLanguageClassifier()
    else:
        offensive_language_classifier = None

    final_job_threads = []
    for run_idx in range(opt['number_of_runs']):
        shared_utils.print_and_log(logging.INFO, f"Launching {run_idx + 1} run........", should_print=True)
        old_mturk_manager = single_run(opt,
                                       bot_agent,
                                       offensive_language_classifier,
                                       pass_qual_id,
                                       fail_qual_id,
                                       dedicated_worker_qual_id,
                                       dedicated_workers_list)
        # Spawn separate threads for previous run manager
        # final settlement(expiring hits, deleting servers)
        thread = Thread(target=run_final_job, args=(old_mturk_manager,))
        thread.daemon = True
        thread.start()
        final_job_threads.append(thread)

    shared_utils.print_and_log(logging.INFO, "Waiting all final jobs to finish", should_print=True)
    for th in final_job_threads:
        th.join()
    shared_utils.print_and_log(logging.INFO, "All runs finished", should_print=True)


if __name__ == '__main__':
    args = setup_args()
    main(args)
