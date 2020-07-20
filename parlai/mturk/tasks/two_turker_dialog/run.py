import os

from parlai.core.params import ParlaiParser
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

    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    mturk_agent_ids = ['PERSON_1', 'PERSON_2']
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids, use_db=False)
    mturk_manager.setup_server()

    def run_onboard(worker):
        world = TwoTurkerDialogOnboardWorld(opt=opt, mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        mturk_manager.start_new_run()
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits()

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
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()