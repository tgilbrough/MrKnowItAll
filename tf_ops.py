import tensorflow as tf

def maxout(inputs, num_units, axis=None):
        """Adds a maxout op which is a max pooling performed in filter/channel
        dimension. This can also be used after fully-connected layers to reduce
        number of features.
        Args:
            inputs: A Tensor on which maxout will be performed
            num_units: Specifies how many features will remain after max pooling at the
            channel dimension. This must be multiple of number of channels.
            axis: The dimension where max pooling will be performed. Default is the
            last dimension.
            outputs_collections: The collections to which the outputs are added.
            scope: Optional scope for name_scope.
        Returns:
            A `Tensor` representing the results of the pooling operation.
        Raises:
            ValueError: if num_units is not multiple of number of features.
        """
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                            'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

def batch_linear(args, output_size, bias, bias_start=0.0, scope=None, name=None):
    """Linear map: concat(W[i] * args[i]), where W[i] is a variable.
    Args:
        args: a 3D Tensor with shape [batch x m x n].
        output_size: int, second dimension of W[i] with shape [output_size x m].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: (optional) Variable scope to create parameters in.
        name: (optional) variable name.
    Returns:
        A 3D Tensor with shape [batch x output_size x n] equal to
        concat(W[i] * args[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args.get_shape().ndims != 3:
        raise ValueError("`args` must be a 3D Tensor")

    shape = args.get_shape()
    m = shape[1].value
    n = shape[2].value
    dtype = args.dtype

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        w_name = "weights_"
        if name is not None: w_name += name
        weights = tf.get_variable(w_name, [output_size, m], dtype=dtype)
        
        X_list = tf.unstack(args, axis=2)
        
        newX_list = [tf.matmul(weights,tf.transpose(x)) for x in X_list]

        res = tf.transpose(tf.stack(newX_list),[2,1,0])

        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            b_name = "biases_"
            if name is not None: b_name += name
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                b_name, [output_size, n],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return tf.add(res, biases)

def _to_3d(tensor):
    if tensor.get_shape().ndims != 2:
        raise ValueError("`tensor` must be a 2D Tensor")
    m, n = tensor.get_shape()
    if m.value is None:
        return tf.reshape(tensor, [-1, n.value, 1])
    return tf.reshape(tensor, [m.value, n.value, 1])

def highway_maxout(hidden_size, pool_size):
    """highway maxout network."""

    def compute(u_t, h, u_s, u_e):
        """Computes value of u_t given current u_s and u_e."""
        # reshape
        u_t = _to_3d(u_t)
        h = _to_3d(h)
        u_s = _to_3d(u_s)
        u_e = _to_3d(u_e)

        # non-linear projection of decoder state and coattention
        state_s = tf.concat([u_t, h, u_s, u_e], axis=1)
        
        r = tf.tanh(batch_linear(state_s, hidden_size, False, name='r'))
        
        res = maxout(batch_linear(r, pool_size, True, name='mm'), 1, axis=1)


        # u_r = tf.concat([u_t, r], axis=1)
        
        # # first maxout
        # m_t1 = batch_linear(u_r, pool_size*hidden_size, True, name='m_1')
        # m_t1 = maxout(m_t1, hidden_size, axis=1)
        
        # # second maxout
        # m_t2 = batch_linear(m_t1, pool_size*hidden_size, True, name='m_2')
        # m_t2 = maxout(m_t2, hidden_size, axis=1)

        # # highway connection
        # mm = tf.concat([m_t1, m_t2], axis=1)
        
        # # final maxout
        # res = maxout(batch_linear(mm, pool_size, True, name='mm'), 1, axis=1)

        return res

    return compute
