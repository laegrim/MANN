
from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers, activations, regularizers, constraints
from keras.layers import RNN
import numpy as np

def _add_inbound_node(layer, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):
        """Internal method to create an inbound node for the layer.
        # Arguments
            input_tensors: list of input tensors.
            output_tensors: list of output tensors.
            input_masks: list of input masks (a mask can be a tensor, or None).
            output_masks: list of output masks (a mask can be a tensor, or None).
            input_shapes: list of input shape tuples.
            output_shapes: list of output shape tuples.
            arguments: dictionary of keyword arguments that were passed to the
                `call` method of the layer at the call that created the node.
        """
        input_tensors = _to_list(input_tensors)
        output_tensors = _to_list(output_tensors)
        input_masks = _to_list(input_masks)
        output_masks = _to_list(output_masks)
        input_shapes = _to_list(input_shapes)
        output_shapes = _to_list(output_shapes)

        # Collect input tensor(s) coordinates.
        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for x in input_tensors:
            if hasattr(x, '_keras_history'):
                inbound_layer, node_index, tensor_index = x._keras_history
                inbound_layers.append(inbound_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers.append(None)
                node_indices.append(None)
                tensor_indices.append(None)

        # Create node, add it to inbound nodes.
        Node(
            layer,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )

        # Update tensor history, _keras_shape and _uses_learning_phase.
        for i in range(len(output_tensors)):
        	print(i)
            output_tensors[i]._keras_shape = output_shapes[i]
            uses_lp = any([getattr(x, '_uses_learning_phase', False) for x in input_tensors])
            uses_lp = getattr(layer, 'uses_learning_phase', False) or uses_lp
            output_tensors[i]._uses_learning_phase = getattr(output_tensors[i], '_uses_learning_phase', False) or uses_lp
            output_tensors[i]._keras_history = (layer,
                                                len(layer.inbound_nodes) - 1,
                                                i)


def __call(layer, inputs, **kwargs):
        """Wrapper around self.call(), for handling internal references.
        If a Keras tensor is passed:
            - We call self._add_inbound_node().
            - If necessary, we `build` the layer to match
                the _keras_shape of the input(s).
            - We update the _keras_shape of every input tensor with
                its new shape (obtained via self.compute_output_shape).
                This is done as part of _add_inbound_node().
            - We update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of _add_inbound_node().
        # Arguments
            inputs: Can be a tensor or list/tuple of tensors.
            **kwargs: Additional keyword arguments to be passed to `call()`.
        # Returns
            Output of the layer's `call` method.
        # Raises
            ValueError: in case the layer is missing shape information
                for its `build` call.
        """
        if isinstance(inputs, list):
            inputs = inputs[:]
        with K.name_scope(layer.name):
            # Handle laying building (weight creating, input spec locking).
            if not layer.built:
                # Raise exceptions in case the input is not compatible
                # with the input_spec specified in the layer constructor.
                layer.assert_input_compatibility(inputs)

                # Collect input shapes to build layer.
                input_shapes = []
                for x_elem in _to_list(inputs):
                    if hasattr(x_elem, '_keras_shape'):
                        input_shapes.append(x_elem._keras_shape)
                    elif hasattr(K, 'int_shape'):
                        input_shapes.append(K.int_shape(x_elem))
                    else:
                        raise ValueError('You tried to call layer "' + self.name +
                                         '". This layer has no information'
                                         ' about its expected input shape, '
                                         'and thus cannot be built. '
                                         'You can build it manually via: '
                                         '`layer.build(batch_input_shape)`')
                if len(input_shapes) == 1:
                    layer.build(input_shapes[0])
                else:
                    layer.build(input_shapes)
                layer.built = True

                # Load weights that were specified at layer instantiation.
                if layer._initial_weights is not None:
                    layer.set_weights(layer._initial_weights)

            # Raise exceptions in case the input is not compatible
            # with the input_spec set at build time.
            layer.assert_input_compatibility(inputs)

            # Handle mask propagation.
            previous_mask = _collect_previous_mask(inputs)
            user_kwargs = copy.copy(kwargs)
            if not _is_all_none(previous_mask):
                # The previous layer generated a mask.
                if has_arg(layer.call, 'mask'):
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            # Handle automatic shape inference (only useful for Theano).
            input_shape = _collect_input_shape(inputs)

            # Actually call the layer, collecting output(s), mask(s), and shape(s).
            output = layer.call(inputs, **kwargs)
            output_mask = layer.compute_mask(inputs, previous_mask)

            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = _to_list(output)
            inputs_ls = _to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = K.identity(x)
                output_ls_copy.append(x)
            if len(output_ls_copy) == 1:
                output = output_ls_copy[0]
            else:
                output = output_ls_copy

            # Inferring the output shape is only relevant for Theano.
            if all([s is not None for s in _to_list(input_shape)]):
                output_shape = layer.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            if not isinstance(output_mask, (list, tuple)) and len(output_ls) > 1:
                # Augment the mask to match the length of the output.
                output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            _add_inbound_node(layer, input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if hasattr(layer, 'activity_regularizer') and layer.activity_regularizer is not None:
                regularization_losses = [layer.activity_regularizer(x) for x in _to_list(output)]
                layer.add_loss(regularization_losses, _to_list(inputs))
        return output

class MANN_LSTM(RNN):
    
    def __init__(self, units, memory, batch_size,
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
                    
            cell = MANN_LSTMCell(units, memory, batch_size,
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

	def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants)

        if initial_state is None and constants is None:
            __call(self, inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = hasattr(additional_inputs[0], '_keras_history')
        for tensor in additional_inputs:
            if hasattr(tensor, '_keras_history') != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = __call(self, full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return __call(self, inputs, **kwargs)
            
    def call(self, inputs, mask=None, training=None, initial_state=None):
            
            self.cell._generate_dropout_mask(inputs, training=training)
            self.cell._generate_recurrent_dropout_mask(inputs, training=training)
            self.cell._generate_controller_dropout_mask(inputs, training=training)
            
            super(MANN_LSTM, self).call(inputs, 
                              mask=mask, 
                              training=training, 
                              initial_state=initial_state)

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
    def memory(self):
    	return self.cell.memory

    @property
    def batch_size(self):
    	return self.cell.batch_size

    def get_config(self):
        config = {'units': self.units,
                  'memory': self.memory,
                  'batch_size': self.batch_size,
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
    def __init__(self, units, memory, batch_size,
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
        self.batch_size = batch_size
        
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

        self.memory = self.add_weight(shape = (self.memory, self.units),
        				name = 'memory',
        				initializer = 'zeros',
        				regularizer = None,
        				trainable = False,
        				constraint = None)

        def controller_initializer(shape, *args, **kwargs):
        	return K.concatenate([
        		initializers.Zeros()((self.batch_size, self.memory.shape[0]), *args, **kwargs),
        		initializers.Ones()((self.batch_size, self.memory.shape[0]), *args, **kwargs),
        		initializers.Zeros()((self.batch_size, self.memory.shape[0]), *args, **kwargs),
        		initializers.Zeros()((self.batch_size, self.memory.shape[0]), *args, **kwargs),
        		])

        self.controller = self.add_weight(shape = (self.batch_size, self.memory.shape[0] * 4),
        	name = 'controller',
        	initializer = controller_initializer,
        	regularizer = None,
        	constraint = None,
        	trainable = False)

        
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
        
        self.controller_wu = self.controller[:, :self.memory.shape[0]]
        self.controller_wlu = self.controller[:, self.memory.shape[0]: self.memory.shape[0] * 2]
        self.controller_wr = self.controller[:, self.memory.shape[0] * 2: self.memory.shape[0] * 3]
        self.controller_ww = self.controller[:, self.memory.shape[0] *3:]
        
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
        nth_smallest = K.reshape(v[:, -n], (self.batch_size, 1))
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

