from keras import layers, models

import odl
import odl.contrib.tensorflow as odl_tf

import numpy as np

class ODLForwardLayer(layers.Layer):
    def __init__(self, odl_layer, output_shape_, **kwargs):
        super().__init__(**kwargs)
        self.odl_layer = odl_layer
        self._output_shape = output_shape_

    def call(self, inputs):
        return self.odl_layer(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self._output_shape, 1)


class ODLAdjointLayer(layers.Layer):
    def __init__(self, odl_layer, size, **kwargs):
        super().__init__(**kwargs)
        self.odl_layer = odl_layer
        self.size = size

    def call(self, inputs):
        return self.odl_layer(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size, self.size, 1)

def build_operator(size, n_proj):
    space = odl.uniform_discr(
        [-size/2, -size/2],
        [size/2, size/2],
        [size, size],
        dtype='float32'
    )

    angle_partition = odl.uniform_partition(0.0, np.pi, n_proj)
    detector_partition = odl.uniform_partition(-size/2, size/2, size)

    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    operator = odl.tomo.RayTransform(space, geometry)

    opnorm = odl.power_method_opnorm(operator)
    operator = (1 / opnorm) * operator

    A_layer = odl_tf.as_tensorflow_layer(operator, 'RayTransform')
    AT_layer = odl_tf.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

    sinogram_shape = operator.range.shape

    return A_layer, AT_layer, sinogram_shape


def conv(x, out_channels):
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    return layers.Conv2D(out_channels, 3, padding='same')(x)

def dual(d, Ap, g, n_dual):
    y = layers.Concatenate()([d, Ap, g])
    y = conv(y, n_dual)
    
    return layers.Add()([y, d])

def primal(d, p, AT_layer, size, n_primal):
    At_d = AT_layer(d[..., 0:1])
    
    y = layers.Concatenate()([p, At_d])
    y = conv(y, n_primal)
    
    return layers.Add()([y, p])

def learned_primal_dual_model(n_proj, size = 128, n_primal = 5, n_dual = 5, n_iter=10):
    A_layer_tf, AT_layer_tf, sinogram_shape = build_operator(size, n_proj)

    A_layer = ODLForwardLayer(A_layer_tf, sinogram_shape)
    AT_layer = ODLAdjointLayer(AT_layer_tf, size)

    g_input = layers.Input((*sinogram_shape, 1))
    p0 = layers.Input((size, size, n_primal))
    d0 = layers.Input((*sinogram_shape, n_dual))

    p_k = p0
    d_k = d0

    Ap = A_layer(p_k[..., 1:2])

    for _ in range(n_iter):
        d_k = dual(d_k, Ap, g_input, n_dual)
        p_k = primal(d_k, p_k, AT_layer, size, n_primal)
        Ap = A_layer(p_k[..., 1:2])

    output = p_k[..., 0:1]

    model = models.Model(
        inputs=[g_input, p0, d0],
        outputs=output,
        name='Learned-Primal-Dual'
    )

    return model