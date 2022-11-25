from collections.abc import Sequence

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import gradient_transform
from pennylane.gradients.general_shift_rules import generate_multishifted_tapes
import warnings

@gradient_transform
def spsb_diff(tape: qml.tape, argnum=None, epsilon=0.01):
    """This is the SPSB differentiator. TODO: We might want to implement
    smoothing here.
    
    """

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the gradient of a tape with no trainable "
            "parameters. If this is unintended, please mark trainable "
            "parameters in accordance with the chosen auto differentiation "
            "framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: qml.math.zeros([tape.output_dim, 0])

    shifts = np.random.choice([-1, 1], size=(len(tape.trainable_params),),
                              requires_grad=False)
    gradient_tapes = generate_multishifted_tapes(
        tape,
        indices = range(len(tape.trainable_params)),
        shifts = [epsilon*shifts, -epsilon*shifts]
    )

    def processing_fn(results):
        if (tape._qfunc_output is not None and
            not isinstance(tape._qfunc_output, Sequence)):
            results = [qml.math.squeeze(res) for res in results]
        f_prime = results[0] - results[1]
        if f_prime.shape == ():
            f_prime = qml.numpy.expand_dims(f_prime, axis=0)
        return qml.numpy.einsum('m,n->mn', f_prime, 1/(2*epsilon*shifts))

    return gradient_tapes, processing_fn
