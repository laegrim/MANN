from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers, activations, regularizers, constraints
from keras.layers import RNN
import numpy as np


class MANN_LSTM(RNN):
    
    def __init__(self, Controller, memory_size,
            usage_decay=.95,
            write_gate_initializer='glorot_uniform',
            write_gate_regularizer=None,
            write_gate_constraint=None,
            # memory_initializer='ones',
            # memory_regularizer=None,
            # memory_constraint=None,
            # read_initializer='zeros',
            # read_regularizer=None,
            # read_constraint=None,
            # least_used_weights_initializer='zeros',
            # least_used_weights_regularizer=None,
            # least_used_weights_constraint=None,
            # usage_weights_initializer='zeros',
            # usage_weights_regularizer=None,
            # usage_weights_constraint=None,
            # read_weights_initializer='zeros',
            # read_weights_regularizer=None,
            # read_weights_constraint=None,
            **kwargs):
                            
        cell = MANN_LSTMCell(Controller, memory_size,
                                 usage_decay=usage_decay,
                                 write_gate_initializer=write_gate_initializer,
                                 write_gate_regularizer=write_gate_regularizer,
                                 write_gate_constraint=write_gate_constraint,
                                 # memory_initializer=memory_initializer,
                                 # memory_regularizer=memory_regularizer,
                                 # memory_constraint=memory_constraint,
                                 # read_initializer=read_weights_initializer,
                                 # read_regularizer=read_regularizer,
                                 # read_constraint=read_constraint,
                                 # least_used_weights_initializer=least_used_weights_initializer,
                                 # least_used_weights_regularizer=least_used_weights_regularizer,
                                 # least_used_weights_constraint=least_used_weights_constraint,
                                 # usage_weights_initializer=usage_weights_initializer,
                                 # usage_weights_regularizer=usage_weights_regularizer,
                                 # usage_weights_constraint=usage_weights_constraint,
                                 # read_weights_initializer=read_weights_initializer,
                                 # read_weights_regularizer=read_weights_regularizer,
                                 # read_weights_constraint=read_weights_constraint,
                                 **kwargs)
        
        super(MANN_LSTM, self).__init__(cell, **kwargs)

    def get_initial_state(self, inputs):

        return self.cell.get_initial_state(inputs)
            
    def reinitialize_nt_weights(self):

        self.cell.reinitialize_nt_weights()

    def get_memory(self):
        return self.cell.get_memory()

    def call(self, inputs, mask=None, training=None, initial_state=None):
            
        self.cell._generate_dropout_mask(inputs, training=training)
        self.cell._generate_recurrent_dropout_mask(inputs, training=training)
            
        return super(MANN_LSTM, self).call(inputs, mask=mask, training=training, initial_state=initial_state)

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
         
        if self._states is None:
            self.states = self.get_initial_state(inputs)
            
        output = super(MANN_LSTM, self).__call__(inputs, initial_state, constants, **kwargs)
        output._keras_shape = (inputs.shape[0], self.Controller.units)
        return output

    @property
    def Controller(self):
        return self.cell.Controller

    @property
    def usage_decay(self):
        return self.cell.usage_decay

    @property
    def memory_size(self):
        return self.cell.memory_size

    @property
    def write_gate_initializer(self):
        return self.cell.write_gate_initializer

    @property
    def write_gate_regularizer(self):
        return self.cell.write_gate_regularizer

    @property
    def write_gate_constraint(self):
        return self.cell.write_gate_constraint

    # @property
    # def memory_initializer(self):
    #     return self.cell.memory_initializer

    # @property
    # def memory_regularizer(self):
    #     return self.cell.memory_regularizer

    # @property
    # def memory_constraint(self):
    #     return self.cell.memory_constraint

    # @property
    # def read_initializer(self):
    #     return self.cell.read_initializer

    # @property
    # def read_regularizer(self):
    #     return self.cell.read_regularizer

    # @property
    # def read_constraint(self):
    #     return self.cell.read_constraint

    # @property
    # def least_used_weights_initializer(self):
    #     return self.cell.least_used_weights_initializer

    # @property
    # def least_used_weights_regularizer(self):
    #     return self.cell.least_used_weights_regularizer

    # @property
    # def least_used_weights_constraint(self):
    #     return self.cell.least_used_weights_constraint

    # @property
    # def usage_weights_initializer(self):
    #     return self.cell.usage_weights_initializer

    # @property
    # def usage_weights_regularizer(self):
    #     return self.cell.usage_weights_regularizer

    # @property
    # def usage_weights_constraint(self):
    #     return self.cell.usage_weights_constraint

    # @property
    # def read_weights_initializer(self):
    #     return self.cell.read_weights_initializer

    # @property
    # def read_weights_regularizer(self):
    #     return self.cell.read_weights_regularizer

    # @property
    # def read_weights_constraint(self):
    #     return self.cell.read_weights_constraint

    def get_config(self):
        config = {'Controller': self.Controller.get_config(),
                  'memory_size': self.memory_size,
                  'usage_decay': self.usage_decay,
                  'write_gate_initializer':initializers.serialize(self.write_gate_initializer),
                  'write_gate_regularizer':regularizers.serialize(self.write_gate_regularizer),
                  'write_gate_constraint':constraints.serialize(self.write_gate_constraint),
                  # 'memory_initializer':initializers.serialize(self.memory_initializer),
                  # 'memory_regularizer':regularizers.serialize(self.memory_regularizer),
                  # 'memory_constraint':constraints.serialize(self.memory_constraint),
                  # 'read_initializer':initializers.serialize(self.read_initializer),
                  # 'read_regularizer':regularizers.serialize(self.read_regularizer),
                  # 'read_constraint':constraints.serialize(self.read_constraint),
                  # 'least_used_weights_initializer':initializers.serialize(self.least_used_weights_initializer),
                  # 'least_used_weights_regularizer':regularizers.serialize(self.least_used_weights_regularizer),
                  # 'least_used_weights_constraint':constraints.serialize(self.least_used_weights_constraint),
                  # 'usage_weights_initializer':initializers.serialize(self.usage_weights_initializer),
                  # 'usage_weights_regularizer':regularizers.serialize(self.usage_weights_regularizer),
                  # 'usage_weights_constraint':constraints.serialize(self.usage_weights_constraint),
                  # 'read_weights_initializer':initializers.serialize(self.read_weights_initializer),
                  # 'read_weights_regularizer':regularizers.serialize(self.read_weights_regularizer),
                  # 'read_weights_constraint':constraints.serialize(self.read_weights_constraint)
                  }
        base_config = super(MANN_LSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class MANN_LSTMCell(Layer):
    def __init__(self, Controller, memory_size,
        usage_decay=.95,
        write_gate_initializer='glorot_uniform',
        write_gate_regularizer=None,
        write_gate_constraint=None,
        # memory_initializer='ones',
        # memory_regularizer=None,
        # memory_constraint=None,
        # read_initializer='zeros',
        # read_regularizer=None,
        # read_constraint=None,
        # least_used_weights_initializer='zeros',
        # least_used_weights_regularizer=None,
        # least_used_weights_constraint=None,
        # usage_weights_initializer='zeros',
        # usage_weights_regularizer=None,
        # usage_weights_constraint=None,
        # read_weights_initializer='zeros',
        # read_weights_regularizer=None,
        # read_weights_constraint=None,
        **kwargs):

        super(MANN_LSTMCell, self).__init__(**kwargs)
        
        self.Controller = Controller
        self.usage_decay = usage_decay
        self.memory_size = memory_size

        self.write_gate_initializer = initializers.get(write_gate_initializer)
        # self.memory_initializer = initializers.get(memory_initializer)
        # self.read_initializer = initializers.get(read_initializer)
        # self.least_used_weights_initializer = initializers.get(least_used_weights_initializer)
        # self.usage_weights_initializer = initializers.get(usage_weights_initializer)
        # self.read_weights_initializer = initializers.get(read_weights_initializer)

        self.write_gate_regularizer = regularizers.get(write_gate_regularizer)
        # self.memory_regularizer = regularizers.get(memory_regularizer)
        # self.read_regularizer = regularizers.get(read_regularizer)
        # self.least_used_weights_regularizer = regularizers.get(least_used_weights_regularizer)
        # self.usage_weights_regularizer = regularizers.get(usage_weights_regularizer)
        # self.read_weights_regularizer = regularizers.get(read_weights_regularizer)

        self.write_gate_constraint = constraints.get(write_gate_constraint)
        # self.memory_constraint = constraints.get(memory_constraint)
        # self.read_constraint = constraints.get(read_constraint)
        # self.least_used_weights_constraint = constraints.get(least_used_weights_constraint)
        # self.usage_weights_constraint = constraints.get(usage_weights_constraint)
        # self.read_weights_constraint = constraints.get(read_weights_constraint)

        self.state_size = tuple([None for i in range(5)]) + self.Controller.cell.state_size
        
    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        weights += self._trainable_weights
        weights += self.Controller.trainable_weights
        return weights
        
    def build(self, input_shape):

        self.write_gate = self.add_weight(shape = (32,),
                                            name = 'write_gate',
                                            initializer = self.write_gate_initializer,
                                            regularizer = self.write_gate_regularizer,
                                            constraint = self.write_gate_constraint)

        # self.memory = self.add_weight(shape = (self.memory_size, self.Controller.units),
        #                                     name = 'memory',
        #                                     initializer = self.memory_initializer,
        #                                     regularizer = self.memory_regularizer,
        #                                     constraint = self.memory_constraint,
        #                                     trainable = False)

        # self.read = self.add_weight(shape = (32, self.Controller.units),
        #                                     name = 'read',
        #                                     initializer = self.read_initializer,
        #                                     regularizer = self.read_regularizer,
        #                                     constraint = self.read_constraint,
        #                                     trainable = False)

        # self.least_used_weights = self.add_weight(shape = (self.memory_size, 32),
        #                                     name = 'least_used_weights',
        #                                     initializer = self.least_used_weights_initializer,
        #                                     regularizer = self.least_used_weights_regularizer,
        #                                     constraint = self.least_used_weights_constraint,
        #                                     trainable = False)

        # self.usage_weights = self.add_weight(shape = (self.memory_size, 32),
        #                                     name = 'usage_weights',
        #                                     initializer = self.usage_weights_initializer,
        #                                     regularizer = self.usage_weights_regularizer,
        #                                     constraint = self.usage_weights_constraint,
        #                                     trainable = False)

        # self.read_weights = self.add_weight(shape = (self.memory_size, 32),
        #                                     name = 'read_weights',
        #                                     initializer = self.read_weights_initializer,
        #                                     regularizer = self.read_weights_regularizer,
        #                                     constraint = self. read_weights_constraint,
        #                                     trainable = False)

        self.memory = K.ones((self.memory_size, self.Controller.units)) * 1e-6
        self.read = K.zeros((32, self.Controller.units))
        self.least_used_weights = K.zeros((self.memory_size, 32))
        self.usage_weights = K.zeros((self.memory_size, 32))
        self.read_weights = K.zeros((self.memory_size, 32))

        controller_input_shape = (input_shape[0], None, input_shape[1] + self.Controller.units)
        self.Controller.build(controller_input_shape)
                    
        self.built = True

    def reinitialize_nt_weights(self):


        print(type(self.memory))

        # self.memory.assign(self.memory_initializer(self.memory.shape))
        # self.read.assign(self.read_initializer(self.read.shape))
        # self.least_used_weights.assign(self.least_used_weights_initializer(self.least_used_weights.shape))
        # self.usage_weights.assign(self.usage_weights_initializer(self.usage_weights.shape))
        # self.read_weights.assign(self.read_weights_initializer(self.read_weights.shape))

        self.memory = K.ones((self.memory_size, self.Controller.units)) * 1e-6
        self.read = K.zeros((32, self.Controller.units))
        self.least_used_weights = K.zeros((self.memory_size, 32))
        self.usage_weights = K.zeros((self.memory_size, 32))
        self.read_weights = K.zeros((self.memory_size, 32))

    def get_memory(self):
        return K.get_value(self.memory)

    def _generate_dropout_mask(self, inputs, training=None):

        if hasattr(self.Controller.cell, "_generate_dropout_mask"):
            template = K.zeros_like(inputs)
            template = K.sum(template, axis=2)
            template = K.expand_dims(template)
            template = K.tile(template, [1, 1, self.Controller.units])
            inputs = K.concatenate([inputs, template])
            self.Controller.cell._generate_dropout_mask(inputs, training=training)

    def _generate_recurrent_dropout_mask(self, inputs, training=None):

        if hasattr(self.Controller.cell, "_generate_recurrent_dropout_mask"):
            template = K.zeros_like(inputs)
            template = K.sum(template, axis=2)
            template = K.expand_dims(template)
            template = K.tile(template, [1, 1, self.Controller.units])
            inputs = K.concatenate([inputs, template])
            self.Controller.cell._generate_recurrent_dropout_mask(inputs, training=training)

    def get_initial_state(self, inputs):

        #input should be (samples, timesteps, input_dim)
        #taken from keras.layers.RNN

        c_initial_states = self.Controller.get_initial_state(K.concatenate([inputs, K.zeros((inputs.shape[0], 1, self.Controller.units))]))

        self.state_size = [state.shape for state in c_initial_states]

        return c_initial_states
            
    def call(self, inputs, states, training=None):

        controller_inputs = K.concatenate([inputs, self.read])
        key_list, controller_states = self.Controller.cell.call(controller_inputs, states, training=training)

        #the write weight is only dependent on last cycles states
        write_weights = K.sigmoid(self.write_gate) * self.read_weights + \
                (1 - K.sigmoid(self.write_gate)) * self.least_used_weights

        #figure out how much each row in memory got used for each sample in batch
        self.usage_weights = self.usage_decay * self.usage_weights + self.read_weights + write_weights #(memory, batch)

        #n is one atm, need to implement a variable number of read heads
        #grab the smallest usage value for each sample in the batch, and the index of that row
        v, i = tf.nn.top_k(K.transpose(self.usage_weights), self.memory_size) #v, i are (batch_size, memory)
        nth_smallest = K.reshape(K.transpose(v)[-1,:], [-1]) #(batch_size)
        i_nth_smallest = K.argmin(nth_smallest) #index of minimum value in nth_smallest array
        nth_smallest = tf.expand_dims(nth_smallest, axis = 1) #(batch_size, 1)
        nth_smallest = tf.tile(nth_smallest, [1, self.memory_size]) #(batch_size, self.memory)
        nth_smallest_i = K.reshape(K.transpose(i)[-1:], [-1]) #(batch_size)
        i = nth_smallest_i[i_nth_smallest] #index of min nth_smallest in c_wu
        nth_smallest_i = K.ones_like(nth_smallest_i) * i #(batch_size) of replicated i
        lt = K.less_equal(self.usage_weights, K.transpose(nth_smallest)) #(memory_size, batch_size)
        self.least_used_weights = K.cast(lt, tf.float32)
        #zero out each sample's least used row in the batch
        zeroing_matrix = tf.one_hot(nth_smallest_i, self.memory_size, on_value = 0., off_value = 1., axis = 0) #(memory, batch)
        #zeroing_matrix should have a zero row
        ones_matrix = K.ones_like(tf.matmul(tf.transpose(zeroing_matrix), tf.ones((self.memory_size, self.Controller.units)))) 
        self.memory = K.dot(zeroing_matrix, ones_matrix) * self.memory
        self.memory = K.dot(write_weights, key_list) + self.memory

        #here we find the cosine similarity between keys and memory rows
        #for each batch's key, and then softmax that to find the read weight
        n_key = K.l2_normalize(key_list, 1) #(batch, units), normed rows
        n_memory = K.l2_normalize(self.memory, 1) #(memory, units), normed rows
        mem_cos_similarity = K.dot(n_key, K.transpose(n_memory)) #(batch, memory)
        self.read_weights = K.softmax(K.transpose(mem_cos_similarity)) #(memory, batch)

         #the read value for each sample is the sum of the weight adjusted memory rows
        read = K.dot(K.transpose(self.read_weights), self.memory) #(batch, memory) x (memory, units) = (batch, units)

        self.read = read
 
        return read, controller_states
