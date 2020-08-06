import os
import time
import yaml
import random

from parlai.core.params import ParlaiParser
from parlai.mturk.core import mturk_utils
from parlai.mturk.tasks.two_turker_dialog.worlds import (
    TwoTurkerDialogWorld
)
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.worlds import (
    QualificationTestOnboardWorld,
    TwoTurkerDialogFallbackBotOnboardWorld,
    InteractParlAIModelWorld
)
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.mturk_manager import MturkManagerWithWaitingPoolTimeout
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.task_config import task_config
from parlai.mturk.tasks.two_turker_dialog_fallback_bot.api_bot_agent import APIBotAgent


def main():
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
    opt = argparser.parse_args()

    task_dir = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(task_dir)
    opt.update(task_config)

    with open(os.path.join(task_dir, 'config.yml')) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    opt.update(cfg)

    mturk_agent_ids = ['CHILD', 'KARU']
    mturk_manager = MturkManagerWithWaitingPoolTimeout(opt=opt,
                                                       mturk_agent_ids=mturk_agent_ids[:1] if opt.get(
                                                           'force_bot') else mturk_agent_ids,
                                                       use_db=True)
    mturk_manager.setup_server()

    if opt.get('qual_test'):
        qual_pass_name = f'{opt["qual_test_qualification"]}Pass'
        qual_pass_desc = (
            'Qualification for a worker correctly completing the '
            'child companion dialog qualification test task.'
        )
        pass_qual_id = mturk_utils.find_or_create_qualification(
            qual_pass_name, qual_pass_desc, opt['is_sandbox']
        )
        print('Created pass qualification: ', pass_qual_id)

        qual_fail_name = f'{opt["qual_test_qualification"]}Fail'
        qual_fail_desc = (
            'Qualification for a worker not correctly completing the '
            'child companion dialog qualification test task.'
        )
        fail_qual_id = mturk_utils.find_or_create_qualification(
            qual_fail_name, qual_fail_desc, opt['is_sandbox']
        )
        print('Created fail qualification: ', fail_qual_id)

    def run_onboard(worker):
        if opt.get('qual_test'):
            world = QualificationTestOnboardWorld(opt=opt,
                                                  mturk_agent=worker,
                                                  pass_qual_id=pass_qual_id,
                                                  fail_qual_id=fail_qual_id)
        else:
            world = TwoTurkerDialogFallbackBotOnboardWorld(opt=opt,
                                                           mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()
        worker.onboard_leave_time = time.time()
        return world.prep_save_data([worker])

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        mturk_manager.start_new_run()
        mturk_manager.ready_to_accept_workers()

        agent_qualifications = []
        if opt.get('qual_test'):
            agent_qualifications.append(
                {
                    'QualificationTypeId': fail_qual_id,
                    'Comparator': 'DoesNotExist',
                    'ActionsGuarded': 'DiscoverPreviewAndAccept',
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
        mturk_manager.create_hits(qualifications=agent_qualifications)

        def check_workers_eligibility(workers):
            return workers

        eligibility_function = {'func': check_workers_eligibility, 'multiple': True}

        def assign_worker_roles(workers):
            roles = random.sample(mturk_agent_ids, len(mturk_agent_ids))
            for index, worker in enumerate(workers):
                worker.id = roles[index % len(roles)]

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            if len(workers) == 1:
                if workers[0].id == 'CHILD':
                    bot_agent_id = 'KARU'
                else:
                    bot_agent_id = 'CHILD'
                bot_agent = APIBotAgent(opt, bot_agent_id)
                world = InteractParlAIModelWorld(opt, workers[0], bot_agent)
            else:
                world = TwoTurkerDialogWorld(opt=opt,
                                             agents=workers)
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
        if opt.get('qual_test') and opt.get("delete_qual_test_qualification"):
            mturk_utils.delete_qualification(pass_qual_id, opt['is_sandbox'])
            mturk_utils.delete_qualification(fail_qual_id, opt['is_sandbox'])
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
