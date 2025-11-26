import zmq
import json
import base64

class DataPublisher:
    def __init__(self, port=5557, topic='model_out'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.topic = topic

    def publish(self, image=None, status="", confidence=0.0):
        msg = {
            'status': status,
            'confidence': confidence
            'image': image if image is not None else ""
        }

        # Send message as JSON string
        self.socket.send_multipart([topic.encode(), json.dumps(msg).encode()])
        print("Published data")
