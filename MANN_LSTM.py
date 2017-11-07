
from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers, activations, regularizers, constraints
from keras.layers import LSTM
import numpy as np

class MANN_LSTM(LSTM):
    
    def __init__(self, units, memory,
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
                usage_decay=.5,
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
                    
            cell = MANN_LSTMCell(units, memory,
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
        
            super(MANN_LSTM, self).super(LSTM, self).__init__(cell,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       stateful=stateful,
                                       unroll=unroll,
                                       **kwargs) 
            
            self.activity_regularizer = regularizers.get(activity_regularizer)
            
    def call(self, inputs, mask=None, training=None, initial_state=None):
            
            self.cell._generate_dropout_mask(inputs, training=training)
            self.cell._generate_recurrent_dropout_mask(inputs, training=training)
            self.cell._generate_controller_dropout_mask(inputs, training=training)
            
            super().call(inputs, 
                              mask=mask, 
                              training=training, 
                              initial_state=initial_state)
        
class MANN_LSTMCell(Layer):
    def __init__(self, units, memory,
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
                usage_decay=.5,
                **kwargs):

        super(MANN_LSTMCell, self).__init__(**kwargs)
        
        self.units = units
        self.memory = memory
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
        self.memory = memory
        
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

        self.memory = K.zeros((memory, units))

        
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
        
        self.controller_wu = K.zeros((32, self.memory.shape[0]))
        self.controller_wlu = K.ones((32, self.memory.shape[0]))
        self.controller_wr = K.zeros((32, self.memory.shape[0]))
        self.controller_ww = K.zeros((32, self.memory.shape[0]))
        
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
            
        self.reads = 1
        
        self.built = True
            
    def call(self, inputs, states, training=None):
        
        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        cont_dp_mask = self._controller_dropout_mask
        
        h_tm1 = states[0]
        c_tm1 = states[1]
        r_tm1 = states[2]
        
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
 
        
        if 0 < self.controller_dropout < 1.:
            
            controller_r = h * cont_dp_mask[0]
            
            
        else:
            
            controller_r = h
            
        #calculate the write weights
        self.controller_ww = K.sigmoid(self.write_gate) * self.controller_wr + \
                    (1 - K.sigmoid(self.write_gate)) * self.controller_wlu
        
        #calculate read weights and retrieve the appropriate memory
        n_controller_r = K.l2_normalize(controller_r, 1)
        n_memory = K.l2_normalize(self.memory, 1)
        t_n_memory = K.transpose(n_memory)
        mem_cos_similarity = K.dot(n_controller_r, t_n_memory)
        self.controller_wr = K.softmax(mem_cos_similarity)
        r = K.dot(self.controller_wr, self.memory)
        self.reads += 1        
        
        #calculate the usage weights
        self.controller_wu = self.usage_decay * self.controller_wu + \
                            self.controller_wr + self.controller_ww
        
        #calculate the least used weights
        v, i = tf.nn.top_k(self.controller_wu, self.controller_wu.shape[1])
        n = min(self.reads, self.memory.shape[1])
        nth_smallest = v[:, -n]
        smallest_index = tf.reduce_min(i[:, -1])
        nth_smallest = tf.matmul(nth_smallest, tf.constant(1., shape=(1, self.memory.shape[0])))
        lt = tf.less_equal(self.controller_wu, nth_smallest)
        self.controller_wlu = tf.cast(lt, tf.float32)
        
        #zero the least used memory location
        #note this is not correct right notw, smallest index is the smallest
        #index of the vector of indicies of smallest values over the batch,
        #not the index of the smallest value over the batch
        zero_array = tf.constant([[1.] if i != smallest_index else [0.] for i in range(self.memory.shape[0])])
        ones_array = tf.ones((1, self.units))
        self.memory = tf.matmul(zero_array, ones_array) * self.memory
        
        #update the memory
        self.memory = tf.matmul(tf.transpose(self.controller_ww), h) + self.memory
        
        
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return r, [h, c, r]

