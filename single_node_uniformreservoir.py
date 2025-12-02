from pathlib import Path

CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
]
searchpaths = [p for p in CANDIDATES if p.exists()]

#tuple of repos
repos = ('machine_learning_library',)

import sys
import os
# In Jupyter notebooks, __file__ is not defined. Use the current working directory instead.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from spnc import spnc_anisotropy
import spnc_ml as ml
from deterministic_mask import fixed_seed_mask, max_sequences_mask

import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from sklearn.metrics import classification_report
from formal_Parameter_Dynamics_Preformance import *


class single_node_uniformreservoir:
    def __init__(self, Nin, Nout, Nvirt, m0=0.01, dilution=1.0, identity = False, res = None, ravel_order = 'C', mask_object = None):
        
        # 1. Mask 初始化逻辑升级
        if mask_object is not None:
            self.M = mask_object
        else:
            self.M = binary_mask(Nin, Nvirt, m0, dilution, identity)
        self.res = res
        self.ravel_order = ravel_order


    def transform(self, x, params, Nthreads=1, **kwargs):
        
        # ======== Nthreads ========
        Nthreads = 1
        # if "Nthreads" in params.keys():
        #     Nthreads = params["Nthreads"]

        # ======== Add mask to input ========
        J = self.M.apply(x)

        # ======== unfold input ========
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

        # ======== transform ========
        if self.res is not None:
            S_1d = self.res(J_1d, params, **kwargs)
        else:
            S_1d = np.copy(J_1d)

        # ======== fold output ========
        if J.dtype == object:
            S = np.copy(J)
            idx = 0
            for i in range(len(S)):
                size = np.prod(J[i].shape) if self.ravel_order is not None else J[i].shape[0]
                S[i] = S_1d[idx:idx+size]
                idx += size
            if self.ravel_order is not None:
                for i,Si in enumerate(S):
                    S[i] = S[i].reshape(J[i].shape, order=self.ravel_order)

        else:
            S = S_1d.reshape(J.shape, order=self.ravel_order) if self.ravel_order is not None else np.copy(S_flat)

        return S, J
