import numpy as np
def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)
 

def mode_dot(tensor, matrix_or_vector, mode):
        """n-mode product of a tensor and a matrix or vector at the specified mode

        Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


        Parameters
        ----------
        tensor : ndarray
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        ndarray
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        multi_mode_dot : chaining several mode_dot in one call
        """
        # the mode along which to fold might decrease if we take product with a vector
        fold_mode = mode
        new_shape = list(tensor.shape)

        if np.ndim(matrix_or_vector) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                    ))
            new_shape[mode] = matrix_or_vector.shape[0]
            vec = False

        elif np.ndim(matrix_or_vector) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                    ))
            if len(new_shape) > 1:
                new_shape.pop(mode)
            else:
                new_shape = [1]
            vec = True

        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                             'Provided array of dimension {} not in [1, 2].'.format(np.ndim(matrix_or_vector)))

        res = np.dot(matrix_or_vector, unfold(tensor, mode))

        if vec: # We contracted with a vector, leading to a vector
            return np.reshape(res, new_shape)
        else: # tensor times vec: refold the unfolding
            return fold(res, fold_mode, new_shape)

