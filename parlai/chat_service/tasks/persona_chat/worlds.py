from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
from parlai.core.agents import create_agent_from_shared


class PersonaChatOnboardWorld(OnboardWorld):
    @staticmethod
    def generate_world(opt, agents):
        return PersonaChatOnboardWorld(opt=opt, agent=agents[0])

    def parley(self):
        self.episodeDone = True


class PersonaChatTaskWorld(World):
    MAX_AGENTS = 1
    MODEL_KEY = 'blender_90M'

    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.episodeDone = False
        self.model = bot
        self.first_time = True

    @staticmethod
    def generate_world(opt, agents):
        if opt['models'] is None:
            raise RuntimeError("Model must be specified")
        return PersonaChatTaskWorld(
            opt,
            agents[0],
            create_agent_from_shared(
                opt['shared_bot_params'][PersonaChatTaskWorld.MODEL_KEY]
            ),
        )

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'ChatbotAgent'

    def parley(self):
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            else:
                print("===act====")
                print(a)
                print("~~~~~~~~~~~")
                if self.first_time and a.get('persona', False):
                    self.model.observe(a)
                    response = {
                        'id': self.model.id,
                        'text': "Successfully set Persona",
                        'episode_done': False
                    }
                    self.agent.observe(response)
                else:
                    self.model.observe(a)
                    response = self.model.act()
                    print("===response====")
                    print(response)
                    print("~~~~~~~~~~~")
                    self.agent.observe(response)
            self.first_time = False

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()


class PersonaChatOverworld(World):
    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return PersonaChatOverworld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def episode_done(self):
        return self.episodeDone

    def parley(self):
        self.episodeDone = True
        return 'default'