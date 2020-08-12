import logging
import json

from parlai.chat_service.services.websocket.sockets import MessageSocketHandler
RESTART_BOT_SERVER_MESSAGE = "[[RESTART_BOT_SERVER_MESSAGE_CRITICAL]]"

class PersonaMessageSocketHandler(MessageSocketHandler):
    def on_message(self, message_text):
        """
        Callback that runs when a new message is received from a client See the
        chat_service README for the resultant message structure.

        Args:
            message_text: A stringified JSON object with a text or attachment key.
                `text` should contain a string message and `attachment` is a dict.
                See `WebsocketAgent.put_data` for more information about the
                attachment dict structure.
        """
        logging.info('websocket message from client: {}'.format(message_text))
        message = json.loads(message_text)
        # Note restart message is case sensitive
        if message.get('text','') == RESTART_BOT_SERVER_MESSAGE:
            self.shutdown_message_callback()
        else:
            message.update({
                'payload': message.get('payload'),
                'sender': {'id': self.sid},
                'recipient': {'id': 0},
            })
            self.message_callback(message)
