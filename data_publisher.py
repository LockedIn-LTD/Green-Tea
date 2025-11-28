import zmq
import json
from datetime import datetime

class DataPublisher:
    def __init__(self, port=5557, topic='model_out'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Use a different port if the subscriber (main.py) connects to a different one,
        # but 5557 is standard for your system based on the main.py PORT variable.
        self.socket.bind(f"tcp://*:{port}")
        self.topic = topic
        print(f"DataPublisher bound to tcp://*:{port} with topic '{topic}'")

    def publish(self, perclos_time_s=0.0):
        msg = {
            'perclos_time_s': perclos_time_s,
            'timestamp': datetime.now().isoformat() # Added timestamp for consistency
        }

        # Send message as JSON string
        self.socket.send_multipart([self.topic.encode(), json.dumps(msg).encode()])
        print(f"[{self.topic}] Perclos: {perclos_time_s:.2f}s")