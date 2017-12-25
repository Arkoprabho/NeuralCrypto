"""
The server side of the communication.
"""
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

    def encode(self, message):
        """
        Encodes the message using the character dictionary.
        Basically converts the characters into integers.
        """
        encoded_data = [self.char_dict[letter] / 10 for letter in message]
        encoded_data = np.asarray(encoded_data).reshape(1, len(encoded_data))
        return encoded_data

    def __prepare_data__(self, message):
        """
        Prepares the data for the encryption process.
        Splits the data into blocks, encodes the data.
        """
        # Break the message down to blocks
        def __break_message__(message, data=None):
            data.append(message[:self.block_size])
            if len(message) < self.block_size:
                last_length = self.block_size - len(data[-1])
                data[-1] += ' ' * last_length
                return data
            return __break_message__(message[self.block_size:], data)

        if len(message) > self.block_size:
            self.broken_data = __break_message__(message)
            # Now this broken data consists of elements that are broken into block size.
        self.input_data = [self.encode(block) for block in self.broken_data]
        # convert this to a numpy array.
        # This is the data that we will be feeding the network
        self.input_data = np.asarray(self.input_data)

    def encrypt(self, message):
        """
        Encrypts the data to be sent
        *Parameters*:
            message: (string) the actual message to be encrypted
        """
        self.__prepare_data__(message)
        input_var = C.input_variable(self.input_data.shape[2], name='Input Sample')
        output_var = C.input_variable(self.input_data.shape[2], name='Output Sample')

        self.model = self.create_model(input_var)
        loss = C.cross_entropy_with_softmax(self.model, output_var)
        label_error = C.classification_error(self.model, output_var)
        learning_rate = 0.01
        lr_schedule = C.learning_parameter_schedule(learning_rate)
        learner = C.sgd(self.model.parameters, lr_schedule)
        trainer = C.Trainer(self.model, [loss, label_error], [learner])
