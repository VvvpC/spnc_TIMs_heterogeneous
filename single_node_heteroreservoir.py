import numpy as np
from spnc import spnc_anisotropy
from joblib import Parallel, delayed
import contextlib
import io




'''
add a new parameter beta_ref 

adjust it by comparing with the single_node_res.py

27/12/24 add a new function for parallel computing

15/01/25 by chen record the signal of each instance and signal of whole reservoir in order to understand the signal magnitude
'''


def parallel_compute_mag(instance, weight, J_1d, params, beta_size_ref):
    with contextlib.redirect_stdout(io.StringIO()):
        result = instance.gen_signal_slow_delayed_feedback_omegaref(
            J_1d, params, beta_size_ref)* weight
    
    return result 


# have no idea why i can't import the binary_mask from mask.py

class binary_mask:
    def __init__(self, Nin, Nvirt, m0=0.1, mask_sparse=1.0, identity=False):
        self.M = 2*m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)

        if identity:
            self.M =np.eye(Nin)

    def apply(self, x):
        if x.dtype == object:
            J = np.copy(x)
            for i,xi in enumerate(x):
                J[i] = np.matmul(xi, self.M.T)
        else:
            J = np.matmul(x, self.M.T)
        return J


class single_node_heteroreservoir:

    def __init__(self, Nin, Nvirt, Nout, m0, beta_prime, beta_size_ref, size_list, dilution = 1.0, identity = False, ravel_order = 'c',**kwargs):

        self.Nvirt = Nvirt
        self.beta_prime = beta_prime
        self.beta_size_ref = beta_size_ref   
        self.m0 = m0
        self.M = binary_mask(Nin, Nvirt, self.m0, dilution, identity)
        self.ravel_order = ravel_order

        self.anisotropy_instances = [
            spnc_anisotropy(0.4, 90, 0, 45, 
            size, 
            restart=True, 
            Primep1=None) 
            for size in size_list
        ]

    def transform(self, x, params, *weights, force_compute=False, nthreads=1):
            """
            Transform function supporting multiple instances and weight combination.
            """
        
            assert len(weights) == len(self.anisotropy_instances), "Weight count should match the number of instances"
            # print('check the weights:', weights)

            Nthreads = 1
            # if "Nthreads" in params.keys():
            #     Nthreads = params["Nthreads"]

            # mask the input signal

            J = self.M.apply(x)

            # ravel the input signal

            if J.dtype == object:
                J_1d = np.copy(J)
                if self.ravel_order is not None:
                    for i,Ji in enumerate(J_1d):
                        J_1d[i] = np.expand_dims(np.ravel(Ji, order=self.ravel_order), axis = -1)
                block_sizes = np.array([ Ji.shape for Ji in J_1d])
                J_1d = np.vstack(J_1d)
            else:
                if self.ravel_order is not None:
                    J_1d = np.expand_dims(np.ravel(J, order=self.ravel_order), axis = -1)
                else:
                    J_1d = np.copy(J)
            
            if Nthreads > 1:
                if J.dtype == object:
                    split_sizes = []
                    for spi in np.array_split(block_sizes, Nthreads):
                        total = 0
                        for si in spi:
                            total += si[0]
                        split_sizes.append(total)

                    params["thread_alloc"] = split_sizes
            
            # transform signals in parallel for each instance

            partial_mags = Parallel(n_jobs=-1)(
                delayed(parallel_compute_mag)(
                    instance, weight, J_1d, params, self.beta_size_ref
                )
                for instance, weight in zip(self.anisotropy_instances, weights)
            )

            # sum the signals from each instance

            S_1d_avarage = np.sum(partial_mags, axis=0)

            # Save the signal of each instance and the signal of the whole reservoir

            instance_info = []
            for i, (instance, weight) in enumerate(zip(self.anisotropy_instances, weights)):
                instance_data = {
                    'weight': weight,
                    'beta_prime': instance.beta_prime,
                    # save the raw signal of each instance
                    'raw_signal': partial_mags[i],
                }
                instance_info.append(instance_data)

            # save the raw signal of the whole reservoir
            reservoir_info = {
                'total_signal': S_1d_avarage
            }

        
            # reshape the signal back to the original shape

            if J.dtype == object:
            
                for i, mag in enumerate(partial_mags):
                    S_instance = np.copy(J)
                    idx = 0
                    for j in range(len(S_instance)):
                        size = (np.product(J[j].shape) if self.ravel_order is not None 
                                else J[j].shape[0])
                        S_instance[j] = mag[idx:idx + size]
                        idx += size
                    
                    if self.ravel_order is not None:
                        for j, Si in enumerate(S_instance):
                            S_instance[j] = Si.reshape(J[j].shape, order=self.ravel_order)
                    instance_info[i]['S_instance'] = S_instance

                else:
                    for i, mag in enumerate(partial_mags):
                        if self.ravel_order is not None:
                            S_instance = mag.reshape(J.shape, order=self.ravel_order)
                        else:
                            S_instance = np.copy(mag)
                        instance_info[i]['S_instance'] = S_instance

            if J.dtype == object:
                S = np.copy(J)
                idx = 0
                for i in range(len(S)):
                    size = np.product(J[i].shape) if self.ravel_order is not None else J[i].shape[0]
                    S[i] = S_1d_avarage[idx:idx+size]
                    idx += size
                if self.ravel_order is not None:
                    for i,Si in enumerate(S):
                        S[i] = S[i].reshape(J[i].shape, order=self.ravel_order)

            else:
                S = S_1d_avarage.reshape(J.shape, order=self.ravel_order) if self.ravel_order is not None else np.copy(S_flat)

            # return final output, masked input signal, information of each instance, containing 'weights', 'beta_prime' and 'raw_signal' and information of the whole reservoir
            return S, J