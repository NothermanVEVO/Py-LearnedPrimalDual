import tensorflow as tf
from keras import layers, models

import odl
import odl.contrib.tensorflow as odl_tf

n_primal = 5
n_dual = 5

size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size], dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space)
operator = odl.tomo.RayTransform(space, geometry)
sinogram_shape = operator.range.shape

opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

A_layer = odl_tf.as_tensorflow_layer(operator, 'RayTransform')
AT_layer = odl_tf.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

def A_op(x):
    return A_layer(x)

def AT_op(x):
    return AT_layer(x)

def A_keras(x):
    return layers.Lambda(lambda t: A_op(t), output_shape=(sinogram_shape[0], sinogram_shape[1], 1))(x)

def AT_keras(x):
    return layers.Lambda(lambda t: AT_op(t), output_shape=(size, size, 1))(x)

def conv(input, out_channels):
    x = layers.Conv2D(32, kernel_size = 3, padding = 'same')(input)
    x = layers.PReLU()(x)

    x = layers.Conv2D(32, kernel_size = 3, padding = 'same')(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(out_channels, kernel_size = 3, padding = 'same')(x)
    
    return x

def dual(d, Ap, g):
    y = layers.Concatenate()([d, Ap, g])

    y = conv(y, n_dual)

    y = layers.Add()([y, d])
    return y

def primal(d, p):
    y = layers.Concatenate()([p, AT_keras(d[..., 0:1])])

    y = conv(y, n_primal)

    y = layers.Add()([y, p])
    return y

def learned_primal_dual_model(input_shape = (sinogram_shape[0], sinogram_shape[1], 1)) -> models.Model: ## input -> sinograma real
    input = layers.Input(input_shape)

    p0 = layers.Input(shape=(size, size, n_primal))
    d0 = layers.Input(shape=(input_shape[0], input_shape[1], n_dual))

    p_k = p0
    d_k = d0

    Ap = A_keras(p_k[..., 1:2])

    for _ in range(10):
        d_k = dual(d_k, Ap, input)
        p_k = primal(d_k, p_k)

        Ap = A_keras(p_k[..., 1:2])

    output = p_k[..., 0:1]

    model = models.Model(inputs=[input, p0, d0], outputs = output, name = 'Learned Primal Dual')
    return model