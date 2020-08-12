from parlai.chat_service.services.persona_chat.agents import PersonaChatAgent
from parlai.chat_service.services.websocket.websocket_manager import WebsocketManager
from parlai.chat_service.services.persona_chat.sockets import PersonaMessageSocketHandler
import logging
import tornado
import time
from tornado.options import options
import gc


class PersonaChatManager(WebsocketManager):
    def _create_agent(self, task_id, socketID):
        return PersonaChatAgent(self.opt, self, socketID, task_id)

    def _make_app(self):
        """
                Starts the tornado application.
                """
        message_callback = self._on_new_message
        shutdown_message_callback = self._on_shutdown_message

        options['log_to_stderr'] = True
        tornado.options.parse_command_line([])

        return tornado.web.Application(
            [
                (
                    r"/websocket",
                    PersonaMessageSocketHandler,
                    {'subs': self.subs, 'message_callback': message_callback,
                     'shutdown_message_callback': shutdown_message_callback},
                )
            ],
            debug=self.debug,
        )
    
    def _on_shutdown_message(self):
        logging.info('sutting down server ...')
        tornado.ioloop.IOLoop.current().stop()

        time.sleep(30)
