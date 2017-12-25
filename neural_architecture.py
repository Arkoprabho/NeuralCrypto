"""
Defines the architecture of the neural network
"""
import string
import cntk as C

# Keep in mind to use a network that can be easily reversed. For example, avoid convolutional
# networks with pooling layers as undoing a pooling operation is not possible.

class Architecture(object):
    """
    Defines the architecture of the neural network to use.
    """
    def __init__(self, seed, block_size):
        """
        Initializes a new instance of the Architecture class
        *Parameters:*
            seed: The seed for the random initializations in the network.
            block_size: The window of inputs to process at a time.
        *Note: * both the parameters should be same between the communicating client and server.
        """
        self.seed = seed
        self.block_size = block_size

        # Create the character dictionary.
        # For the uppercase characters
        self.char_dict = dict(
            zip(
                string.ascii_uppercase, [ord(c)-91 for c in string.ascii_uppercase]
            )
        )
        # Now for the lower letters.
        self.char_dict.update(
            dict(
                zip(
                    string.ascii_lowercase, [ord(c)-96 for c in string.ascii_lowercase]
                )
            )
        )
        # Lets not forget the space shall we.
        self.char_dict[' '] = 0

    def create_model(self, features):
        """
        The architecture of the neural network.
        Here we create an ANN with 2 hidden layers and an output layer
        """
        with C.layers.default_options(init=C.glorot_uniform(seed=self.seed), activation=C.tanh):
            layer = C.layers.Dense(self.block_size, name='Hidden Layer 1')(features)
            layer = C.layers.Dense(self.block_size, name='Hidden Layer 2')(layer)
            return C.layers.Dense(self.block_size, activation=None, name='Output')(layer)
