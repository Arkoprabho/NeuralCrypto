"""
Describes the client side of the communication
"""
import cntk as C
import numpy as np
from neural_architecture import Architecture

class Client(Architecture):
    """
    Acts as the client side of the communication
    """
    def __init__(self, seed, block_size):
        print('Initializing client...')
        super().__init__(seed, block_size)
        self.inverted_char_dict = dict([[v,k] for k,v in self.char_dict.items()])

        # Create the model
        self.input_var = C.input_variable(self.block_size, name='Input Sample')
        self.output_var = C.input_variable(self.block_size, name='Output Sample')
        self.model = self.create_model(self.input_var)

    def decrypt(self, message):
        """
        Decrypts the message sent by the server according to the DN algorithm
        """
        decrypted_data = ''
        input_data = []
        # The message received here is in the form of a list of nd arrays.
        # Typically of the shape (batches, 1, block_size)
        
        # Perform validation here. Check whether the block size is the same
        if np.shape(message)[2] != self.block_size:
            raise Exception('Different block size received. Sync server and client')
        if np.shape(message)[1] != 1:
            raise Exception("Something's wrong. And it cant figure out what!")

        # Now comes the actual decryption part.
        # Steps involved:
            # 1. Invert all the operations that took place to get the decrypted message
            # 2. Once the message is received, make a forward pass and update the weights.
        # Because we set the seed to be the same, we can be assured that the initialization will result in the same matrix.
        
        # Get the layers from the model
        hl1 = self.model.find_by_name('Hidden Layer 1')
        hl2 = self.model.find_by_name('Hidden Layer 2')
        ol = self.model.find_by_name('Output')

        # Reversing the operations.
        for item in message:
            temp = ((((
                # Reverting output layer
                (item - ol.b.value) @ np.linalg.inv(ol.W.value)
                # Reverting hidden layer 2
                ) - hl2.b.value) @ np.linalg.inv(hl2.W.value)
                # Reverting hidden layer 1
                ) - hl1.b.value) @ np.linalg.inv(hl1.W.value)
            # Round the data to 1 decimals and multiply by 10. This is done to remove the rounding errors that are caused
            temp = np.round(temp, decimals=1)[0]
            final_output = np.asarray((temp * 10), dtype=int)
            input_data.append(final_output / 10)

            # Print the data
            for i in final_output:
                decrypted_data += self.inverted_char_dict[i]
        
        input_data = np.asarray(input_data)
        # This is now the same dataset that the network should initially be trained with
        input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
        output_data = np.flip(input_data, axis=2)

        batch_size = 1
        num_batches = (input_data.shape[0] * input_data.shape[1]) / batch_size

        loss = C.losses.squared_error(self.model, self.output_var)
        learning_rate = 0.01
        learner = C.learners.sgd(self.model.parameters, learning_rate)
        trainer = C.train.Trainer(self.model, [loss], [learner])
        input_map = {self.input_var: input_data, self.output_var: output_data}

        
        # Make passes through the network and update the weights
        for i in range(int(num_batches)):
            trainer.train_minibatch(input_map)
        return decrypted_data