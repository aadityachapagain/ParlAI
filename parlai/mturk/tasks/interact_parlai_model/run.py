import os

from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.interact_parlai_model.worlds import (
    InteractParlAIModelOnboardWorld,
    InteractParlAIModelWorld
)
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.interact_parlai_model.task_config import task_config


def main():
    argparser = ParlaiParser(False, True, 'Interactive chat with a model in Mturk')
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()

    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    mturk_agent_id = ['PERSON']
    bot_agent_id = 'BOT'
    bot_agent = create_agent(opt, requireModelExists=True)

    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_id, use_db=True)
    mturk_manager.setup_server()

    def run_onboard(worker):
        world = InteractParlAIModelOnboardWorld(opt=opt, mturk_agent=worker)
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
                worker.id = mturk_agent_id[index % len(mturk_agent_id)]

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):

            shared_bot_agent = create_agent_from_shared(bot_agent.share())
            shared_bot_agent.id = bot_agent_id

            world = InteractParlAIModelWorld(opt=opt,
                                             mturk_agent=workers[0],
                                             parlai_agent=shared_bot_agent)
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
