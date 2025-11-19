# ZMQ Video Client Library
# This file contains the ZMQVideoReceiver class, designed to replace cv2.VideoCapture
# for programs that subscribe to a frame stream published by camera_driver.py.

import numpy as np
import zmq

class ZMQVideoReceiver:
    """
    A class that mimics cv2.VideoCapture but receives frames via ZeroMQ.
    It connects to a specified ZMQ Publisher address and provides frames
    using the familiar cap.read() interface.
    """
    def __init__(self, address="tcp://127.0.0.1:5555"):
        """
        Initializes the ZMQ connection.
        
        Args:
            address (str): The ZMQ address and port of the camera driver, 
                           e.g., "tcp://127.0.0.1:5555"
        """
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.is_connected = False
        
        # Connect to the driver and subscribe to all topics
        try:
            self.socket.connect(self.address)
            # The empty string means "subscribe to all messages"
            self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
            
            # Use Poller to manage non-blocking reads and timeouts
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
            print(f"[ZMQ Video Receiver] Connected to stream at {self.address}")
            self.is_connected = True
        except zmq.error.ZMQError as e:
            print(f"[ERROR] ZMQ connection failed to {self.address}. Is the driver running? Error: {e}")
            self.is_connected = False

    def isOpened(self):
        """Returns True if the ZMQ socket is connected and ready to receive."""
        return self.is_connected

    def read(self):
        """
        Mimics cap.read() -> returns (ret, frame)
        
        It polls the socket for data, waits for a short duration, and
        reconstructs the frame from the received raw bytes.
        """
        # Poll the socket, waiting up to 50 milliseconds for a message to arrive
        socks_ready = dict(self.poller.poll(50))

        if self.socket in socks_ready:
            try:
                # 1. Receive Metadata (blocking for the first part of the message)
                md = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                
                # 2. Receive the raw image data
                frame_data = self.socket.recv(flags=zmq.NOBLOCK)
                
                # 3. Reconstruct the NumPy array from the metadata and raw bytes
                # This is the fast part of the zero-copy reconstruction
                frame = np.frombuffer(frame_data, dtype=np.dtype(md['dtype'])).reshape(md['shape'])
                
                return True, frame
            except zmq.Again:
                # The socket had data ready but the read operation timed out or was interrupted
                return False, None
            except Exception as e:
                # Handle unexpected corruption or socket errors
                print(f"[ERROR] Failed to read/reconstruct frame from ZMQ: {e}")
                return False, None
        
        # If the poller timed out (50ms elapsed), no frame was available
        return False, None

    def release(self):
        """Cleans up the ZMQ context and socket, mimicking cap.release()."""
        if self.is_connected:
            print("[ZMQ Video Receiver] Releasing resources.")
            self.socket.close()
            self.context.term()