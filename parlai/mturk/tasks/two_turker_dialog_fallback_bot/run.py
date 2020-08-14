import os
import time
import yaml
import random
import requests
import logging
from threading import Thread

from parlai.core.params import ParlaiParser
from parlai.mturk.core import mturk_utils
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.worlds import (
    QualificationTestOnboardWorld,
    InteractParlAIModelWorld
)
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.mturk_manager import MturkManagerWithWaitingPoolTimeout
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.task_config import task_config
import parlai.mturk.core.shared_utils as shared_utils
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.api_bot_agent import APIBotAgent


def setup_args():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--bot-host',
        dest='bot_host',
        required=True,
        help='IP Address of the bot API'
    )
    argparser.add_argument(
        '--bot-port',
        dest='bot_port',
        default=8000,
        help='Port address of Bot agent'
    )
    argparser.add_argument(
        '--bot-username',
        dest='bot_username',
        required=True,
        help='username to use for authentication in bot API'
    )
    argparser.add_argument(
        '--bot-password',
        dest='bot_password',
        required=True,
        help='password to use for authentication in bot API'
    )
    parsed_args = argparser.parse_args()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_args['task'] = os.path.basename(task_dir)
    parsed_args.update(task_config)

    with open(os.path.join(task_dir, 'config.yml')) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    parsed_args.update(cfg)

    return parsed_args


def create_qualification(opt):
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
                               f"Created Pass Qualification {pass_qual_id} and fail qualification {fail_qual_id}")
    return pass_qual_id, fail_qual_id


def single_run(opt):
    mturk_agent_ids = ['CHILD', 'KARU']
    mturk_manager = MturkManagerWithWaitingPoolTimeout(opt=opt,
                                                       mturk_agent_ids=[random.choice(mturk_agent_ids)],
                                                       use_db=True)
    mturk_manager.setup_server()

    pass_qual_id, fail_qual_id = create_qualification(opt)

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
            # {
            #     'QualificationTypeId': '3USHAHCQKTJI4JTX5CNFEU1GP95GAN',
            #     'Comparator': 'GreaterThanOrEqualTo',
            #     'IntegerValues': opt['min_hit_approval_rate'],
            #     'ActionsGuarded': 'DiscoverPreviewAndAccept'
            # }
        ]
        if not opt['is_sandbox']:
            agent_qualifications.extend([
                {
                    'QualificationTypeId': '00000000000000000071',
                    'Comparator': 'In',
                    'LocaleValues': [{'Country': country} for country in opt['allowed_countries']],
                    'ActionsGuarded': 'DiscoverPreviewAndAccept',
                },
            ])
        mturk_manager.create_hits(qualifications=agent_qualifications)

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
            bot_agent = APIBotAgent(opt, bot_agent_id, mturk_manager.task_group_id)
            world = InteractParlAIModelWorld(opt, workers[0], bot_agent)

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
    except BaseException:
        raise
    finally:
        if opt.get("delete_qual_test_qualification"):
            mturk_utils.delete_qualification(pass_qual_id, opt['is_sandbox'])
            mturk_utils.delete_qualification(fail_qual_id, opt['is_sandbox'])
            shared_utils.print_and_log(logging.INFO, "Deleted qualification..............")
        return mturk_manager


def run_final_job(manager):
    shared_utils.print_and_log(logging.INFO, f"Running Final Job of run {manager.task_group_id}", should_print=True)
    manager.shutdown()


def main(opt):
    final_job_threads = []
    for run_idx in range(opt['number_of_runs']):
        shared_utils.print_and_log(logging.INFO, "Sending restart instruction....", should_print=True)
        # requests.post(f'http://{opt["bot_host"]}:{str(opt["bot_port"])}/interact',
        #               json={'text': '[[RESTART_BOT_SERVER_MESSAGE_CRITICAL]]'},
        #               auth=(opt['bot_username'],
        #                     opt['bot_password'])
        #               )
        # time.sleep(opt['sleep_between_runs'])
        shared_utils.print_and_log(logging.INFO, f"Launching {run_idx + 1} run........", should_print=True)
        old_mturk_manager = single_run(opt)
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
