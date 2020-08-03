import logging
from parlai.chat_service.services.websocket.agents import WebsocketAgent


class PersonaChatAgent(WebsocketAgent):
    def put_data(self, message):
        """
        Put data into the message queue.

        Args:
            message: dict. An incoming websocket message. See the chat_services
                README for the message structure.
        """
        logging.info(f"Received new message: {message}")
        message.update({
            'episode_done': False,
            'payload': message.get('payload')
        })

        self._queue_action(message, self.action_id)
        self.action_id += 1
