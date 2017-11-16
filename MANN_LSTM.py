from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers, activations, regularizers, constraints
from keras.layers import RNN
import numpy as np


class MANN_LSTM(RNN):
    
    def __init__(self, Controller, memory_size,
            usage_decay=.95,
            stateful=True,
            write_gate_initializer='glorot_uniform',
            write_gate_regularizer=None,
            write_gate_constraint=None,
            **kwargs):
        
        self.stateful = stateful
                    
        cell = MANN_LSTMCell(Controller, memory_size,
                                 usage_decay=usage_decay,
                                 write_gate_initializer=write_gate_initializer,
                                 write_gate_regularizer=write_gate_regularizer,
                                 write_gate_constraint=write_gate_constraint,
                                 **kwargs)
        
        super(MANN_LSTM, self).__init__(cell, stateful=stateful, **kwargs)

    def get_initial_state(self, inputs):

        return self.cell.get_initial_state(inputs)
            
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
    
    def reset_states(self, states=None):

        if not self.stateful:
            raise AttributeError('Layer must be stateful.')

        batch_size = self.input_spec[0].shape[0]
        mock_input = K.zeros((batch_size, 1, 1))

        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        # initialize state if None
        if self.states is None or self.states[0] is None:
            self.get_initial_state(mock_input)
        elif states is None:
            self.get_initial_state(mock_input)
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != (batch_size, dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)
                
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

    def get_config(self):
        config = {'Controller': self.Controller.get_config(),
                  'memory_size': self.memory_size,
                  'usage_decay': self.usage_decay,
                  #'write_gate_initializer':initializers.serialize(self.write_gate_initializer),
                  'write_gate_regularizer':regularizers.serialize(self.write_gate_regularizer),
                  'write_gate_constraint':constraints.serialize(self.write_gate_constraint)
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
        **kwargs):

        super(MANN_LSTMCell, self).__init__(**kwargs)
        
        self.Controller = Controller
        self.usage_decay = usage_decay
        self.memory_size = memory_size
        self.write_gate_initializer = write_gate_initializer
        self.write_gate_regularizer = write_gate_regularizer
        self.write_gate_constraint = write_gate_constraint
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

        controller_input_shape = (input_shape[0], None, input_shape[1] + self.Controller.units)
        self.Controller.build(controller_input_shape)
                    
        self.built = True

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

        template = K.zeros_like(inputs)
        template = K.sum(template, axis=(1,2)) #(samples, )
        template = K.expand_dims(template) #(samples, 1)
        z_temp = K.transpose(K.zeros_like(template))
        template = K.tile(template, [1, self.Controller.units]) #(samples, units)

        r_tm1 = K.zeros_like(template)
        m_tm1 = K.ones((self.memory_size, self.Controller.units), dtype=tf.float32) * 1e-6
        c_wu_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)
        c_wlu_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)
        c_wr_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)

        c_initial_states = self.Controller.get_initial_state(K.concatenate([inputs, K.expand_dims(r_tm1, axis=1)]))

        self.state_size = [r_tm1.shape,
                            m_tm1.shape,
                            c_wu_tm1.shape,
                            c_wlu_tm1.shape,
                            c_wr_tm1.shape] + \
                            [state.shape for state in c_initial_states]

        return [r_tm1, m_tm1, c_wu_tm1, c_wlu_tm1, c_wr_tm1] + \
                [state for state in c_initial_states]
            
    def call(self, inputs, states, training=None):

        r_tm1 = states[0]
        m_tm1 = states[1]
        c_wu_tm1 = states[2]
        c_wlu_tm1 = states[3]
        c_wr_tm1 = states[4]

        controller_states = states[5:]
        controller_inputs = K.concatenate([inputs, r_tm1])
        key_list, controller_states = self.Controller.cell.call(controller_inputs, controller_states, training=training)

        #the write weight is only dependent on last cycles states
        c_ww = K.sigmoid(self.write_gate) * c_wr_tm1 + \
                (1 - K.sigmoid(self.write_gate)) * c_wlu_tm1

        #here we find the cosine similarity between keys and memory rows
        #for each batch's key, and then softmax that to find the read weight
        n_key = K.l2_normalize(key_list, 1) #(batch, units), normed rows
        n_memory = K.l2_normalize(m_tm1, 1) #(memory, units), normed rows
        mem_cos_similarity = K.dot(n_key, K.transpose(n_memory)) #(batch, memory)
        c_wr = K.softmax(K.transpose(mem_cos_similarity)) #(memory, batch)

        #the read value for each sample is the sum of the weight adjusted memory rows
        read = K.dot(K.transpose(c_wr), m_tm1) #(batch, memory) x (memory, units) = (batch, units)

        #figure out how much each row in memory got used for each sample in batch
        c_wu = self.usage_decay * c_wu_tm1 + c_wr + c_ww #(memory, batch)

        #n is one atm, need to implement a variable number of read heads
        #grab the smallest usage value for each sample in the batch, and the index of that row
        v, i = tf.nn.top_k(K.transpose(c_wu), self.memory_size) #v, i are (batch_size, memory)
        nth_smallest = K.reshape(K.transpose(v)[-1,:], [-1]) #(batch_size)
        i_nth_smallest = K.argmin(nth_smallest) #index of minimum value in nth_smallest array
        nth_smallest = tf.expand_dims(nth_smallest, axis = 1) #(batch_size, 1)
        nth_smallest = tf.tile(nth_smallest, [1, self.memory_size]) #(batch_size, self.memory)
        nth_smallest_i = K.reshape(K.transpose(i)[-1:], [-1]) #(batch_size)
        i = nth_smallest_i[i_nth_smallest] #index of min nth_smallest in c_wu
        nth_smallest_i = K.ones_like(nth_smallest_i) * i #(batch_size) of replicated i
        lt = K.less_equal(c_wu, K.transpose(nth_smallest)) #(memory_size, batch_size)
        c_wlu = K.cast(lt, tf.float32)
        #zero out each sample's least used row in the batch
        zeroing_matrix = tf.one_hot(nth_smallest_i, self.memory_size, on_value = 0., off_value = 1., axis = 0) #(memory, batch)
        #zeroing_matrix should have a zero row
        ones_matrix = K.ones_like(tf.matmul(tf.transpose(zeroing_matrix), tf.ones((self.memory_size, self.Controller.units)))) 
        memory = K.dot(zeroing_matrix, ones_matrix) * m_tm1
        memory = K.dot(c_ww, key_list) + memory
 
        return read, [read, memory, c_wu, c_wlu, c_wr] + controller_states
