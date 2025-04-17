import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from algos.dqn_base import DQNAgent

class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent: splits into separate value and advantage streams after CNN,
    then combines them to produce final Q-values, improving learning efficiency.
    """
    def _build_model(self):
        """Builds a dueling CNN architecture."""
        # Input layer expects shape (H, W, C)
        input_layer = Input(shape=self.state_shape)
        # Shared convolutional layers
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        # Value stream
        v = Dense(512, activation='relu')(x)
        v = Dense(1, activation='linear')(v)
        # Advantage stream
        a = Dense(512, activation='relu')(x)
        a = Dense(self.action_size, activation='linear')(a)
        # Combine value and advantage into Q-values
        def combine(inputs):
            value, adv = inputs
            adv_mean = tf.reduce_mean(adv, axis=1, keepdims=True)
            return value + (adv - adv_mean)
        q_values = Lambda(combine)([v, a])
        # Create and compile model
        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
