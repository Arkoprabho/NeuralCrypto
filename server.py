"""
The server side of the communication.
"""
import string
import cntk as C
import numpy as np
from neural_architecture import Architecture

# The client and the server will have same neural network architecture.

class Server(Architecture):
    """
    The server side of the communication.
    """
    def __init__(self, server_seed, server_block_size):
        """
        Initializes a new instance of the Server.
        """
        print('Initializing server....')
        Architecture.__init__(self, seed=server_seed, block_size=server_block_size)
        
    def encrypt(self, message):
        """
        Encrypts the data to be sent
        *Parameters*:
            message: (string) the actual message to be encrypted
        """
        # Break the message down to blocks
        