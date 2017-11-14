from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers, activations, regularizers, constraints
from keras.layers import RNN
import numpy as np


class MANN_LSTM(RNN):
    
    def __init__(self, units, memory_size,
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                controller_dropout=0.,
                usage_decay=.95,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
           
            if K.backend() == 'cntk':
                if not kwargs.get('unroll') and (dropout > 0 or recurrent_dropout > 0):
                    warnings.warn(
                        'RNN dropout is not supported with the CNTK backend '
                        'when using dynamic RNNs (i.e. non-unrolled). '
                        'You can either set `unroll=True`, '
                        'set `dropout` and `recurrent_dropout` to 0, '
                        'or use a different backend.')
                    dropout = 0.
                    recurrent_dropout = 0.
                    
            cell = MANN_LSTMCell(units, memory_size,
                        activation = activation,
                        recurrent_activation = recurrent_activation,
                        use_bias = use_bias,
                        kernel_initializer = kernel_initializer,
                        recurrent_initializer = recurrent_initializer,
                        unit_forget_bias = unit_forget_bias,
                        bias_initializer = bias_initializer,
                        kernel_regularizer = kernel_regularizer,
                        recurrent_regularizer = recurrent_regularizer,
                        bias_regularizer = bias_regularizer,
                        kernel_constraint = kernel_constraint,
                        recurrent_constraint = recurrent_constraint,
                        bias_constraint = bias_constraint,
                        dropout = dropout,
                        recurrent_dropout = recurrent_dropout,
                        controller_dropout = controller_dropout,
                        usage_decay=usage_decay,
            **kwargs)
        
            super(MANN_LSTM, self).__init__(cell,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       stateful=stateful,
                                       unroll=unroll,
                                       **kwargs)
            
            self.activity_regularizer = regularizers.get(activity_regularizer)

    def get_initial_state(self, inputs):

        return self.cell.get_initial_state(inputs)
            
    def call(self, inputs, mask=None, training=None, initial_state=None):
            
            self.cell._generate_dropout_mask(inputs, training=training)
            self.cell._generate_recurrent_dropout_mask(inputs, training=training)
            self.cell._generate_controller_dropout_mask(inputs, training=training)
            
            return super(MANN_LSTM, self).call(inputs, 
                              mask=mask, 
                              training=training, 
                              initial_state=initial_state)

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):

        output = super(MANN_LSTM, self).__call__(inputs, initial_state, constants, **kwargs)
        output._keras_shape = (inputs.shape[0], self.units)
        return output
    
    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def controller_dropout(self):
        return self.cell.controller_dropout

    @property
    def usage_decay(self):
        return self.cell.usage_decay

    @property
    def memory_size(self):
        return self.cell.memory_size

    def get_config(self):
        config = {'units': self.units,
                  'memory': self.memory,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'controller_dropout': self.controller_dropout,
                  'usage_decay': self.usage_decay,
                  }
        base_config = super(MANN_LSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class MANN_LSTMCell(Layer):
    def __init__(self, units, memory_size,
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                controller_dropout=0.,
                usage_decay=.95,
                **kwargs):

        super(MANN_LSTMCell, self).__init__(**kwargs)
        
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.write_gate_initializer = initializers.get('zero')
        self.unit_forget_bias = unit_forget_bias
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.write_gate_regularizer = regularizers.get(None)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.write_gate_constraint = constraints.get(None)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.controller_dropout = min(1., max(0., controller_dropout))

        self.state_size = (self.units, self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self._controller_dropout_mask = None

        self.usage_decay = usage_decay
        self.memory_size = memory_size
        
    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(4)]
        else:
            self._dropout_mask = None

    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(4)]
        else:
            self._recurrent_dropout_mask = None
 
    def _generate_controller_dropout_mask(self, inputs, training=None):
        
        if 0 < self.controller_dropout < 1:
            ones = K.ones((self.units, self.memory.shape[0]))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._controller_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range (3)]

        else:
            self._controller_dropout_mask = None
                
    def build(self, input_shape):
        
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(shape = (input_dim, self.units * 4),
                                     name = 'kernel',
                                     initializer = self.kernel_initializer,
                                     regularizer = self.kernel_regularizer,
                                     constraint = self.kernel_constraint)
        
        self.recurrent_kernel = self.add_weight(shape = (self.units, self.units * 5),
                                               name = 'recurrent_kernel',
                                               initializer = self.recurrent_initializer,
                                               regularizer = self.recurrent_regularizer,
                                               constraint = self.recurrent_constraint)
        
        self.write_gate = self.add_weight(shape = (1,),
                                            name = 'write_gate',
                                            initializer = self.write_gate_initializer,
                                            regularizer = self.write_gate_regularizer,
                                            constraint = self.write_gate_constraint)
        
        if self.use_bias:
            
            if self.unit_forget_bias:
                
                def bias_initializer(shape, *args, **kwargs):
                    
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                
                bias_initializer = self.bias_initializer
                
            self.bias = self.add_weight(shape = (self.units * 4,),
                                       name = 'bias',
                                       initializer = bias_initializer,
                                       regularizer = self.bias_regularizer,
                                       constraint = self.bias_constraint)

        else:
            
            self.bias = None
                
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]
        
        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3: self.units * 4]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units * 4:] 
        
        if self.use_bias:
            
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
            
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
                    
        self.built = True
       
    def get_initial_state(self, inputs):

        #input should be (samples, timesteps, input_dim)
        #taken from keras.layers.RNN
        template = K.zeros_like(inputs)
        template = K.sum(template, axis=(1,2)) #(samples, )
        template = K.expand_dims(template) #(samples, 1)
        z_temp = K.transpose(K.zeros_like(template))
        template = K.tile(template, [1, self.units]) #(samples, units)

        h_tm1 = K.zeros_like(template)
        c_tm1 = K.zeros_like(template)
        r_tm1 = K.zeros_like(template)
        m_tm1 = K.ones((self.memory_size, self.units), dtype=tf.float32)
        c_wu_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)
        c_wlu_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)
        c_wr_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)
        c_ww_tm1 = K.dot(K.zeros((self.memory_size, 1)), z_temp)

        self.state_size = [h_tm1.shape,
                            c_tm1.shape,
                            r_tm1.shape,
                            m_tm1.shape,
                            c_wu_tm1.shape,
                            c_wlu_tm1.shape,
                            c_wr_tm1.shape,
                            c_ww_tm1.shape]

        return [h_tm1, c_tm1, r_tm1, m_tm1, 
                c_wu_tm1, c_wlu_tm1, c_wr_tm1,
                c_ww_tm1]
            
    def call(self, inputs, states, training=None):

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        cont_dp_mask = self._controller_dropout_mask

        h_tm1 = states[0]
        c_tm1 = states[1]
        r_tm1 = states[2]
        m_tm1 = states[3]
        c_wu_tm1 = states[4]
        c_wlu_tm1 = states[5]
        c_wr_tm1 = states[6]
        c_ww_tm1 = states[7]
        
        if 0 < self.dropout < 1.:
            
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
            
        else:
            
            inputs_i = inputs_f = inputs_c = inputs_o = inputs
            
        x_i = K.dot(inputs_i, self.kernel_i)
        x_f = K.dot(inputs_f, self.kernel_f)
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = K.dot(inputs_o, self.kernel_o)
        
        if self.use_bias:
            
            x_i = K.bias_add(x_i, self.bias_i)
            x_f = K.bias_add(x_f, self.bias_f)
            x_c = K.bias_add(x_c, self.bias_c)
            x_o = K.bias_add(x_o, self.bias_o)
            
        if 0 < self.recurrent_dropout < 1.:
            
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
                    
        else:
            
            h_tm1_i = h_tm1_f = h_tm1_c = h_tm1_o = h_tm1
            
        h_tm1_i = K.dot(h_tm1_i, self.recurrent_kernel_i)
        h_tm1_f = K.dot(h_tm1_f, self.recurrent_kernel_f)
        h_tm1_c = K.dot(h_tm1_c, self.recurrent_kernel_c)
        h_tm1_o = K.dot(h_tm1_o, self.recurrent_kernel_o)
        
        #memories are fed back as input next cycle
        r_tm1_i = K.dot(r_tm1, self.recurrent_kernel_r)
            
        i = self.recurrent_activation(x_i + h_tm1_i + r_tm1_i)
        f = self.recurrent_activation(x_f + h_tm1_f)
        c = f * c_tm1 + i * self.activation(x_c + h_tm1_c)
        o = self.recurrent_activation(x_o + h_tm1_o)
        h = o * self.activation(c)

        #get the key, check if there's dropouts
        #key_list is (batch_size, units), one key for each sample in batch
        if 0 < self.controller_dropout < 1.:
            
            key_list = h * cont_dp_mask[0] 
            
        else:
            
            key_list = h

        #we want (keys, batches) so we can figure out read weights 
        #for each sample in the batch
        key_list = K.transpose(key_list) #(units, batch_size)

        #the write weight is only dependent on last cycles states
        c_ww = K.sigmoid(self.write_gate) * c_wr_tm1 + \
                (1 - K.sigmoid(self.write_gate)) + c_wlu_tm1

        #here we find the cosine similarity between keys and memory rows
        #for each batch's key, and then softmax that to find the read weight
        n_key = K.l2_normalize(key_list, 0) 
        n_memory = K.l2_normalize(m_tm1, 1)
        mem_cos_similarity = K.dot(n_memory, n_key)
        c_wr = K.softmax(mem_cos_similarity)

        #the read value for each sample is the sum of the weight adjusted memory rows
        read = K.dot(tf.transpose(m_tm1), c_wr) #(units, memory) x (memory, samples)
        read = K.transpose(read) #(samples, units)

        #figure out how much each row in memory got used for each sample in batch
        c_wu = self.usage_decay * c_wu_tm1 + c_wr + c_ww

        #grab the smallest usage value for each sample in the batch, and the index of that row
        v, i = tf.nn.top_k(tf.transpose(c_wu), self.memory_size) #v, i are (batch_size, memory)
        nth_smallest = K.reshape(K.transpose(v)[-1,:], [-1]) #(batch_size)
        nth_smallest = tf.expand_dims(nth_smallest, axis = 1) #(batch_size, 1)
        nth_smallest = tf.tile(nth_smallest, [1, self.memory_size]) #(batch_size, self.memory)
        nth_smallest_i = K.reshape(K.transpose(i)[-1:], [-1]) #(batch_size)
        lt = K.less_equal(c_wu, K.transpose(nth_smallest))
        c_wlu = K.cast(lt, tf.float32)
        #zero out each sample's least used row in the batch
        zeroing_vector = tf.one_hot(nth_smallest_i, self.memory_size, on_value = 0., off_value = 1., axis = 0) #(memory, batch)
        ones_vector = K.ones_like(tf.matmul(tf.transpose(zeroing_vector), tf.ones((self.memory_size, self.units)))) 
        memory = K.dot(zeroing_vector, ones_vector) * m_tm1
        memory = K.dot(c_ww, tf.transpose(key_list)) + memory

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
         
        return read, [h, c, read, memory, c_wu, c_wlu, c_wr, c_ww]
