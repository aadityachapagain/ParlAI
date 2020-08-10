from parlai.chat_service.services.persona_chat.agents import PersonaChatAgent
from parlai.chat_service.services.websocket.websocket_manager import WebsocketManager
from parlai.chat_service.services.persona_chat.sockets import PersonaMessageSocketHandler
import tornado
from tornado.options import options


class PersonaChatManager(WebsocketManager):
    def _create_agent(self, task_id, socketID):
        return PersonaChatAgent(self.opt, self, socketID, task_id)

    def _make_app(self):
        """
                Starts the tornado application.
                """
        message_callback = self._on_new_message

        options['log_to_stderr'] = True
        tornado.options.parse_command_line([])

        return tornado.web.Application(
            [
                (
                    r"/websocket",
                    PersonaMessageSocketHandler,
                    {'subs': self.subs, 'message_callback': message_callback},
                )
            ],
            debug=self.debug,
        )