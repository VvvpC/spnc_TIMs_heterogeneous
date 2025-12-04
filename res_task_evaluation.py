# libraries
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from single_node_heteroreservoir import single_node_heteroreservoir as shr
from joblib import Parallel, delayed
from contextmanager import no_print
import contextlib
import io

CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
    Path(r"/Users/vvvp./Desktop"),
   

]
searchpaths = [p for p in CANDIDATES if p.exists()]
#tuple of repos
repos = ('machine_learning_library',)

# Add local modules and paths to local repos
from deterministic_mask import fixed_seed_mask, max_sequences_mask
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from audio_preprocess import mfcc
from sklearn.metrics import classification_report

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from res_manager import ResManager

# 创建单独温度下的任务函数，用于并行计算
def narma10_single_temperature(temp, res_manager, net, x_test, y_test, spacer):
    with contextlib.redirect_stdout(io.StringIO()):
        S_test,_ = res_manager.transform(x_test, env_temp=temp)
        pred = net.forward(S_test)
        error = MSE(pred, y_test)
        predNRMSE = NRMSE(pred, y_test, spacer=spacer)
        return {
            'temp': temp,
            'MSE': error,
            'NRMSE': predNRMSE,
            'y_test': y_test,
            'pred': pred
        }
def ti46_single_temperature(Nout, temp, res_manager, net, x_test, test_label,nblocks):
    with contextlib.redirect_stdout(io.StringIO()):
        S_test,_ = res_manager.transform(x_test, env_temp=temp)
        
        post_process = block_process
        post_process = lambda S, *args, **kwargs: np.copy(S)
        
        z_test = post_process(S_test, nblocks, plot=False)

        conf_mat = np.zeros((Nout, Nout))
        Ncorrect = 0
        for i in range(len(z_test)):
            pi = net.forward(z_test[i])
            pl = np.argmax(np.mean(pi, axis=0))
            conf_mat[test_label[i], pl] += 1.0
            if pl == test_label[i]:
                Ncorrect += 1
        final_accuracy = Ncorrect/len(S_test)
        return {
            'temp': temp,
            'accuracy': final_accuracy,
            'conf_mat': conf_mat,
        }


class TaskEvaluation:
    def __init__(self, res_manager: ResManager):
        self.res_manager = res_manager
        self.task_configs = res_manager.task_configs
        self.temp_configs = res_manager.temp_configs

        # [关键修改] 初始化时不加载数据，只设为 None (懒加载)
        # NARMA
        self.x_train_narma = None
        self.y_train_narma = None
        self.x_test_narma = None
        self.y_test_narma = None
        self.net_narma = None
        
        # TI-46
        self.xntrain_ti46 = None
        self.train_label_ti46 = None
        self.xntest_ti46 = None
        self.test_label_ti46 = None
        self.split1_ti46 = None
        self.net_ti46 = None


    def generate_narma_data(self):
        if self.net_narma is not None:
            return
        u, d = NARMA10(self.task_configs.narma_train_len + self.task_configs.narma_test_len, seed=self.task_configs.narma_seed)

        self.x_train_narma = u[:self.task_configs.narma_train_len]
        self.y_train_narma = d[:self.task_configs.narma_train_len]
        self.x_test_narma = u[self.task_configs.narma_train_len:]
        self.y_test_narma = d[self.task_configs.narma_train_len:]

        self.net_narma = self.get_narma_weights_at_training_temperature()

    def get_narma_weights_at_training_temperature(self):

        # Net setup
        Nin = self.x_train_narma[0].shape[-1]
        Nout = 1

        net = linear(Nin, Nout, bias=self.res_manager.params_configs.bias)

        # confirm the temperature is fixed
        train_temp = self.temp_configs.beta_temp_ref

        # Training
        S_train, _ = self.res_manager.transform(
            self.x_train_narma, 
            env_temp=train_temp
            )
        np.size(S_train)
        RR.Kfold_train(
            net, 
            S_train, 
            self.y_train_narma, 
            10, 
            quiet = True, 
            seed_training=self.task_configs.narma_seed
            )

        return net

    def testing_narma_at_varying_temperatures(self, n_jobs=-1):

        if self.net_narma is None:
            self.generate_narma_data()

        print(f"Testing NARMA10 at varying temperatures (Parallel n_jobs={n_jobs})...")

        # Testing at varying temperatures
        temp_list = self.temp_configs.gen_temp_list()
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(narma10_single_temperature)(
                temp, 
                self.res_manager, 
                self.net_narma, 
                self.x_test_narma, 
                self.y_test_narma, 
                self.task_configs.spacer_NRMSE)
            for temp in tqdm(temp_list, desc = 'NARMA10 Temp Sweep'))

        results.sort(key=lambda x: x['temp'])

        return results

    def prepare_ti46_data(self):
        if self.net_ti46 is not None:
            return

        print("Preparing TI46 data...")
        (self.xntrain_ti46, self.train_label_ti46, self.xntest_ti46, self.test_label_ti46, self.split1_ti46, self.split2_ti46) = self._gen_ti46_data()

        self.net_ti46 = self.get_ti46_weights_at_training_temperature()

    def _gen_ti46_data(self):
        
        speakers = self.task_configs.ti46_speakers

        self.train_signal_ti46, self.train_label_ti46, self.train_rate_ti46, self.train_speaker_ti46 = TI46.load_TI20(speakers, digits_only=True, train=True)
        self.test_signal_ti46, self.test_label_ti46, self.test_rate_ti46, self.test_speaker_ti46 = TI46.load_TI20(speakers, digits_only=True, train=False)

        def stratified_split(labels, N, seed=1234):
            indexes = tuple(np.unique(x) for x in labels)
            sizes = tuple(len(x) for x in indexes)
            label_ids = list(np.array([], dtype=int) for i in range(np.prod(sizes)))

            for i, label in enumerate(zip(*labels)):
                ids = np.array([np.where(l == index)[0][0] for l, index in zip(label, indexes)])
                idx = np.ravel_multi_index(ids, sizes)
                label_ids[idx] = np.append(label_ids[idx], i)

            rng = np.random.default_rng(seed)
            split1 = np.array([], dtype=int)
            split2 = np.array([], dtype=int)
            for idxs in label_ids:
                rng.shuffle(idxs)
                split1 = np.append(split1, idxs[:N])
                split2 = np.append(split2, idxs[N:])
            rng.shuffle(split1)
            rng.shuffle(split2)
            return split1, split2 
        
        self.split1_ti46, self.split2_ti46 = stratified_split((self.train_speaker_ti46, self.train_label_ti46), 9)

        # Pre-processing
        from audio_preprocess import mfcc_func
        pre_process = mfcc_func
        nf = self.task_configs.ti46_nfft

        self.xntrain_ti46 = pre_process(self.train_signal_ti46, self.train_rate_ti46, nfft=nf)
        self.xntrain_ti46 = normalise(self.xntrain_ti46)

        self.xntest_ti46 = pre_process(self.test_signal_ti46, self.test_rate_ti46, nfft=nf)
        self.xntest_ti46 = normalise(self.xntest_ti46)

        return self.xntrain_ti46, self.train_label_ti46, self.xntest_ti46, self.test_label_ti46, self.split1_ti46, self.split2_ti46


    def get_ti46_weights_at_training_temperature(self):

        Nin = self.xntrain_ti46[0].shape[-1]
        self.Nout = len(np.unique(self.train_label_ti46))

        # confirm the temperature is fixed
        train_temp = self.temp_configs.beta_temp_ref

        # Training
        S_train, _ = self.res_manager.transform(
            self.xntrain_ti46, 
            env_temp=train_temp
            )
        
        post_process = block_process
        post_process = lambda S, *args, **kwargs: np.copy(S)

        Nblocks = self.task_configs.ti46_nblocks
        z_train = post_process(S_train, Nblocks, plot=False)

        y_train_1h = create_1hot_like(self.Nout, z_train, self.train_label_ti46)

        net = linear(self.res_manager.params_configs.Nvirt, self.Nout, bias=self.res_manager.params_configs.bias)

        z_train_flat = np.vstack(z_train[self.split1_ti46])
        y_train_1h_flat = np.vstack(y_train_1h[self.split1_ti46])

        RR.Kfold_train(
            net, 
            z_train_flat, 
            y_train_1h_flat, 
            5,
            quiet = True, 
            seed_training=self.task_configs.ti46_seed
            )
        
        conf_mat = np.zeros((self.Nout, self.Nout))
        pred_labels = np.zeros(len(self.split2_ti46), dtype=int)
        Ncorrect = 0
        for i, (zi, li) in enumerate(zip(z_train[self.split2_ti46], self.test_label_ti46[self.split2_ti46])):
            pi = net.forward(zi)
            pl = np.argmax(np.mean(pi, axis=0))
            pred_labels[i] = pl
            conf_mat[li, pl] += 1.0
            if pl == li:
                Ncorrect += 1

        return net  

    def testing_ti46_at_varying_temperatures(self, n_jobs=-1):

        if self.net_ti46 is None:
            self.prepare_ti46_data()

        temp_list = self.temp_configs.gen_temp_list()
        nblocks = self.task_configs.ti46_nblocks

        print(f"Testing TI46 at varying temperatures (Parallel n_jobs={n_jobs})...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(ti46_single_temperature)(
                self.Nout,
                temp, 
                self.res_manager, 
                self.net_ti46, 
                self.xntest_ti46, self.test_label_ti46, nblocks)
            for temp in tqdm(temp_list, desc = 'TI46 Temp Sweep'))
        results.sort(key=lambda x: x['temp'])
        return results















