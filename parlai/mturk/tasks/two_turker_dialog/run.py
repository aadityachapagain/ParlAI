import os
import yaml

from parlai.core.params import ParlaiParser
from parlai.mturk.core import mturk_utils
from parlai.mturk.tasks.two_turker_dialog.worlds import (
    TwoTurkerDialogOnboardWorld,
    TwoTurkerDialogWorld
)
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.two_turker_dialog.task_config import task_config


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()

    task_dir = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(task_dir)
    opt.update(task_config)

    with open(os.path.join(task_dir, 'config.yml')) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    opt.update(cfg)

    mturk_agent_ids = ['PERSON_1', 'PERSON_2']
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids, use_db=True)
    mturk_manager.setup_server()

    qual_pass_name = 'ChildCompanionDialogQualificationPass'
    qual_pass_desc = (
        'Qualification for a worker correctly completing the '
        'child companion dialog qualification test task.'
    )
    pass_qual_id = mturk_utils.find_or_create_qualification(
        qual_pass_name, qual_pass_desc, opt['is_sandbox']
    )
    print('Created pass qualification: ', pass_qual_id)

    qual_fail_name = 'ChildCompanionDialogQualificationFail'
    qual_fail_desc = (
        'Qualification for a worker not correctly completing the '
        'child companion dialog qualification test task.'
    )
    fail_qual_id = mturk_utils.find_or_create_qualification(
        qual_fail_name, qual_fail_desc, opt['is_sandbox']
    )
    print('Created fail qualification: ', fail_qual_id)

    def run_onboard(worker):
        world = TwoTurkerDialogOnboardWorld(opt=opt,
                                            mturk_agent=worker,
                                            pass_qual_id=pass_qual_id,
                                            fail_qual_id=fail_qual_id)
        while not world.episode_done():
            world.parley()
        world.shutdown()

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
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
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
        mturk_utils.delete_qualification(pass_qual_id, opt['is_sandbox'])
        mturk_utils.delete_qualification(fail_qual_id, opt['is_sandbox'])
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
