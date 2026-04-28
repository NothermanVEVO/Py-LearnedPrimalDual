from keras import layers, models

import odl
import odl.contrib.tensorflow as odl_tf


def build_operator(size):
    space = odl.uniform_discr(
        [-size/2, -size/2],
        [size/2, size/2],
        [size, size],
        dtype='float32'
    )

    geometry = odl.tomo.parallel_beam_geometry(space)
    operator = odl.tomo.RayTransform(space, geometry)

    opnorm = odl.power_method_opnorm(operator)
    operator = (1 / opnorm) * operator

    A_layer = odl_tf.as_tensorflow_layer(operator, 'RayTransform')
    AT_layer = odl_tf.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

    sinogram_shape = operator.range.shape

    return A_layer, AT_layer, sinogram_shape


def A_keras(x, A_layer, sinogram_shape):
    return layers.Lambda(
        lambda t: A_layer(t),
        output_shape=(sinogram_shape[0], sinogram_shape[1], 1)
    )(x)


def AT_keras(x, AT_layer, size):
    return layers.Lambda(
        lambda t: AT_layer(t),
        output_shape=(size, size, 1)
    )(x)


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
    At_d = AT_keras(d[..., 0:1], AT_layer, size)
    
    y = layers.Concatenate()([p, At_d])
    y = conv(y, n_primal)
    
    return layers.Add()([y, p])

def learned_primal_dual_model(size = 128, n_primal = 5, n_dual = 5, n_iter=10):
    A_layer, AT_layer, sinogram_shape = build_operator(size)

    g_input = layers.Input((*sinogram_shape, 1))
    p0 = layers.Input((size, size, n_primal))
    d0 = layers.Input((*sinogram_shape, n_dual))

    p_k = p0
    d_k = d0

    Ap = A_keras(p_k[..., 1:2], A_layer, sinogram_shape)

    for _ in range(n_iter):
        d_k = dual(d_k, Ap, g_input, n_dual)
        p_k = primal(d_k, p_k, AT_layer, size, n_primal)
        Ap = A_keras(p_k[..., 1:2], A_layer, sinogram_shape)

    output = p_k[..., 0:1]

    model = models.Model(
        inputs=[g_input, p0, d0],
        outputs=output,
        name='Learned Primal Dual'
    )

    return model