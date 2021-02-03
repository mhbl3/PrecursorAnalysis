import pandas as pd
import numpy as np
from ..dataModel.dataProcessing import DataContainer, myDataset
import matplotlib.pyplot as plt
from torch import nn
import torch as tc
from sklearn.metrics import *
from tqdm import tqdm
from torch.nn import functional as F
from ..utils import one_hot_embedding, window_padding
import pickle as pkl
import copy 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings

myseed = 45237552
np.random.seed(120)
tc.manual_seed(myseed)

# Model Container Class Definition
class ModelContainer():
    def __init__(self, **kwargs):
        if "name" in kwargs:
            self.name = kwargs["name"]
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = tc.device("cuda:0" if tc.cuda.is_available else "cpu")
        print(f"Model container using {self.device}")
        if "optimizer" in kwargs:
            self.optimizer = kwargs["optimizer"]
        else:
            self.optimizer = "adam"
        
        if "data" in kwargs:
            assert type(kwargs["data"]) == DataContainer
            self.myData = kwargs["data"]
   
    def printModel(self):
        print(self)
    
    def show_feature_importance(self, columns, flight_id=0,
                                adjust_proba_window=False,
                                remove_constant=False,
                                class_interest = None,
                                show_largest_change= False,
                                plot_save = None,
                                plot = True,figsize= (30,30)):
        """
        To be used only when a precursor subclass was created

        Parameters
        ----------
        columns : list of columns names
        flight_id : int, optional
            id of flight of interest. The default is 0 (only one flight).
        figsize : set, optional
            width and length of canvas. The default is (30,30).
        remove_constant : bool, optional
            removes features that had a constant value. The default is False
        Returns
        -------
        None.

        """
        if "cuda" in self.device:
            proba_time = self.proba_time.cpu().detach().numpy()
            precursor_proba = self.precursor_proba.cpu().detach().numpy()
        else:
            proba_time = self.proba_time.detach().numpy()
            precursor_proba = self.precursor_proba.detach().numpy()
        if adjust_proba_window:
            #FIXME fix this issue
            proba_time = window_padding(proba_time[flight_id,:].flatten(), self.x_train.shape[1])
        else:
            if (class_interest is None) or (self.n_classes==1):
                proba_time = proba_time[flight_id, :]
            else:
                #TODO: Check shape
                if (len(proba_time.shape) ==3) and (class_interest is not None):
                    proba_time = proba_time[flight_id, :, class_interest]
                elif (len(proba_time.shape) ==3) and (class_interest is None):
                    raise ValueError("class_interest is not set")
        time_step_masks = np.where(proba_time > self.threshold)[0]
        diff_time_step_masks = np.diff(time_step_masks)
        where_jump = np.where(diff_time_step_masks > 1)[0] +1
        # self.first_flag_index = []
        self.multiple_precursors = [True if where_jump.shape[0] == 0 else False][0]

        # Search where precursor proba > threshold and obtain feature precursor score
        if where_jump.shape[0] == 0:
            time_step_mask = time_step_masks
            self.first_flag_index = time_step_mask[0]
            temp = precursor_proba[flight_id, time_step_mask, :]
            if show_largest_change:
                precursor_proba_val_to_plot = np.zeros(temp.shape[-1])
                for feature in range(temp.shape[-1]):
                    precursor_proba_val_to_plot[feature] = abs(temp[0, feature]- temp[-1, feature])
            else:
                precursor_proba_val_to_plot = np.average(abs(temp-0.5), axis=0)
    
            sorted_avg = list(np.argsort(precursor_proba_val_to_plot)[::-1])
    
            if remove_constant:
                for i in range(precursor_proba.shape[-1]):
                    if np.std(precursor_proba[flight_id, time_step_mask, i]) <= 1e-4:
                        sorted_avg.remove(i)
            if plot:
                plt.figure(figsize=figsize)
    
                plt.barh(range(len(sorted_avg)), precursor_proba_val_to_plot[sorted_avg][::-1])
                plt.yticks(range(len(sorted_avg)), columns[sorted_avg][::-1])
                plt.grid(True)
                plt.show()
                if plot_save is not None:
                    plt.savefig(plot_save, dpi=600)

                
            self.sorted_features = columns[sorted_avg].values
            self.sorted_features_values = precursor_proba_val_to_plot[sorted_avg]
            self.list_sorted_features = np.nan
            self.list_sorted_features_values = np.nan
        else:
            split_time_masks = np.split(time_step_masks, where_jump)
            self.list_sorted_features = []
            self.list_sorted_features_values = []
            for time_step_mask in split_time_masks:
                self.first_flag_index = time_step_mask[0]
                temp = precursor_proba[flight_id, time_step_mask, :]

                if show_largest_change:
                    precursor_proba_val_to_plot = np.zeros(temp.shape[-1])
                    for feature in range(temp.shape[-1]):
                        precursor_proba_val_to_plot[feature] = abs(temp[0, feature]- temp[-1, feature])
                else:
                    precursor_proba_val_to_plot = np.average(abs(temp-0.5), axis=0)
        
                sorted_avg = list(np.argsort(precursor_proba_val_to_plot)[::-1])
        
                if remove_constant:
                    for i in range(precursor_proba.shape[-1]):
                        if np.std(precursor_proba[flight_id, time_step_mask, i]) <= 1e-4:
                            sorted_avg.remove(i)
                            
                self.list_sorted_features.append(columns[sorted_avg].values)
                self.list_sorted_features_values.append(precursor_proba_val_to_plot[sorted_avg])
                self.sorted_features_values = np.nan
                self.sorted_features = np.nan
                if plot:
                    plt.figure(figsize=figsize)
        
                    plt.barh(range(len(sorted_avg)), precursor_proba_val_to_plot[sorted_avg][::-1])
                    plt.yticks(range(len(sorted_avg)), columns[sorted_avg][::-1])
                    plt.grid(True)
                    plt.show()
                    if plot_save is not None:
                        plt.savefig(plot_save, dpi=600)

    
        
    def save(self, filename):
        """
        Save model as a .pt file
        :param filename: str
            Location of directory and name of the file to be saved
        :return:
        """
        if "pt" not in filename:
            filename = filename + ".pt"
        with open(filename, "wb") as f:
            tc.save(self, f)
        print(f"Model Saved! (path: {filename})")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def plot_feature_effects(self, full_length, columns,
                            flight_id=0, save_path=None,
                            class_interest = None,
                            show_precursor_range=False,
                            rescaling_factor = 0, **kw):
        # Initializations
         ticks_on = kw.get("ticks_on", True)
         counter = 0
         width = 4*5.5
         if "cuda" in self.device:
            proba_time = self.proba_time.cpu().detach().numpy()
            precursor_proba = self.precursor_proba.cpu().detach().numpy()
         else:
            proba_time = self.proba_time.detach().numpy()
            precursor_proba = self.precursor_proba.detach().numpy()
            
         if rescaling_factor < 0:
             precursor_proba = precursor_proba + rescaling_factor
         elif rescaling_factor > 0:
             precursor_proba = abs(precursor_proba-rescaling_factor)
         else:
             precursor_proba = precursor_proba - rescaling_factor
                  
         n_features = precursor_proba.shape[2]
         if class_interest is not None:
             grey_area_class_plot_idx = class_interest
         else:
             grey_area_class_plot_idx = 0
             # if len(proba_time.shape) == 3:
             #    proba_time = proba_time[flight_id, :, class_interest]
         height = 3.5* int(n_features/4+1)
         fig, ax1 = plt.subplots(int(n_features/4+1), 4, figsize=(width, height))
         fig.tight_layout(pad=6.5)
             
         for i in range(0, int(n_features-4)+1):
             for j in range(0, 4):
                 if i == 0 and j == 0:
                     if len(proba_time.shape) < 3:
                         num_loops = 1
                     else:
                         num_loops = proba_time.shape[-1]
                     for class_idx in range(num_loops):
                         tmp_proba_time = proba_time[flight_id,:] if len(proba_time.shape) < 3 else proba_time[flight_id,:, class_idx]
                         if proba_time.shape[1] != full_length:
                             # pad values
                             r_proba_time = self.window_padding(tmp_proba_time.flatten(),
                                                                full_length)
                         else:
                            r_proba_time = tmp_proba_time.flatten()

                         if (show_precursor_range) and (grey_area_class_plot_idx==class_idx):
                             mask_idx = np.where(r_proba_time > self.threshold)[0]
                             diff_time_step_masks = np.diff(mask_idx)
                             where_jump = np.where(diff_time_step_masks > 1)[0] +1
                         if class_idx == 0:
                            ax1[i,j].plot(r_proba_time, "r")
                         else:
                             ax1[i, j].plot(r_proba_time)
                         ax1[i,j].set_ylabel("Probability")
                         ax1[i,j].set_title("Precursor Score")
                         ax1[i,j].grid(True)
                         ax1[i,j].set_xlabel("Distance to event")
                         ax1[i,j].set_yticks(np.arange(0, 1.1, 0.1))
                         if ticks_on:
                             x = np.arange(20 , -0.25, -0.25)
                             ax1[i,j].set_xticks(range(0, full_length, 10))

                         if (show_precursor_range) and (grey_area_class_plot_idx==class_idx):
                             if where_jump.shape[0] == 0:
                                 ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')
                             else:
                                 mask_idxs = mask_idx
                                 split_time_masks = np.split(mask_idxs, where_jump)
                                 for mask_idx in split_time_masks:
                                     ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')
                         if ticks_on:
                            ax1[i,j].set_xticklabels(x[::10])

                     continue
                 if counter == n_features:
                     break
                 # In the case the window used does not match the flight length
                 if  precursor_proba.shape[1] != full_length:
                     # pad values
                     precursor_value = self.window_padding(precursor_proba[flight_id,
                                                            :, counter].flatten(),
                                                           full_length)
                 else:
                     precursor_value = precursor_proba[flight_id,
                                                            :, counter].flatten()
           
                 ax1[i,j].plot(precursor_value, "r")
                 ax1[i,j].set_title(f"Feature: {columns[counter]}")
                 ax1[i,j].set_xlabel("Distance Remaining")
                 ax1[i,j].set_ylabel("Probabilities")
                 ax1[i,j].grid(True)
                 ax1[i,j].set_yticks(np.arange(0, 1.1, 0.1))
                 if ticks_on:
                     x = np.arange(20 , -0.25, -0.25)
                     ax1[i,j].set_xticks(range(0, full_length, 10))
                     ax1[i,j].set_xticklabels(x[::10])
                 if show_precursor_range:
                     if where_jump.shape[0] == 0:
                          ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')
                     else:
                          for mask_idx in split_time_masks:
                              ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')
                 counter += 1

         if save_path is not None:
             text = save_path+"//precursor_proba.pdf"
             plt.savefig(text, dpi=600)
        
    def window_padding(self, data_to_pad, full_length, method="interp"):
        """
        Used to adjust the size of the windows created by CNN outputs
        
        Parameters
        ----------
        data_to_pad : numpy array
            1D array containing the data to pad.
        full_length : int
            final length to be used.
        method : str, optional
            methodology to use "interp" or "steps". The default is "interp".
        
        Returns
        -------
        numpy array
            1D array of requested length (full_length).
            
        """
        # seq_len = int(round(full_length/data_to_pad.shape[0]))
        if method == "interp":
            x = np.linspace(0, full_length-1, full_length)
            xp = np.linspace(0, full_length-1, data_to_pad.shape[0]) #might need to be -n instead of 1
            fp = data_to_pad
            return np.interp(x=x, xp=xp, fp=fp)
        elif method =="steps":
            temp = np.zeros((full_length, 1))
            seq_len = int(round(full_length/data_to_pad.shape[0]))
            for n, m in enumerate(range(0, temp.shape[0], seq_len)):
                try:
                    temp[m:m+seq_len] = data_to_pad[n]
                except:
                    temp[m:m+seq_len] = data_to_pad[-1]
            return temp


    def train_Precursor_binary(self, clf, X_train, y_train,
                         X_val=None, y_val=None, l2=0,
                         num_epochs=200, learning_rate=0.01, verbose=0,
                         model_out_cpu = True, **kw):
      
        # Convert to parameters CUDA Tensors
        clf = clf.to(self.device)

        self.n_epochs = num_epochs
        print_every_epochs = kw.pop("print_every_epochs", 10)
        # Binary cross entropy loss, learning rate and l2 regularization
        weight = kw.pop("class_weight", None)
        if weight is not None:
            weight = tc.Tensor(weight).to(self.device)
        if len(np.unique(y_train)) <= 2:
            criterion = tc.nn.BCELoss(weight=weight)
            self.task = "binary"
        else:
            criterion = tc.nn.CrossEntropyLoss(weight=weight)
            self.task = "multiclass"
            self.n_classes = len(np.unique(y_train))
        
        print("Classification task: {}".format(self.task))
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam(clf.parameters(),
                                  lr=learning_rate, weight_decay=l2)
        else:
            optimizer = tc.optim.SGD(clf.parameters(),
                                      lr=learning_rate, weight_decay=l2)
        
        # Init loss history and balanced accuracy
        hist = np.zeros(num_epochs)
        val_hist = np.zeros(num_epochs)
        b_acc = np.zeros(num_epochs)
        val_b_acc = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)
        val_f1 = np.zeros(num_epochs)
          
        # Conversion to tensors
        if not tc.is_tensor(X_train):
            X_train = tc.Tensor(X_train)
        if not tc.is_tensor(y_train):
            y_train = tc.Tensor(y_train.flatten())
            if self.task == "multiclass":
                y_train = y_train.type(tc.int64)
            
        if X_val is not None:
            if not tc.is_tensor(X_val):
                X_val = tc.Tensor(X_val)
            if not tc.is_tensor(y_val):
                y_val = tc.Tensor(y_val)
            data_val = myDataset(X_val, y_val)
      
        if self.batch_size is None:
            self.batch_size = X_train.size(0)
            warnings.warn("Setting the batch size = full training set could overwhelm GPU memory")
        
        data_train = myDataset(X_train, y_train)
        if self.use_stratified_batch_size is False:
            print("Mini-batch strategy: Random sampling")
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        else:
            print("Mini-batch strategy: Stratified")
            # get class counts
            weights = []
            for label in tc.unique(y_train):
                count = len(tc.where(y_train == label)[0])
                weights.append(1 / count)
            weights = tc.tensor(weights).to(self.device)
            samples_weights = weights[y_train.type(tc.LongTensor).to(self.device)]
            sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, sampler=sampler)
        if X_val is not None:
                dataloader_val = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)
        # Train the model
        try:
            for epoch in tqdm(range(num_epochs)):
                batch_acc = []
                batch_val_acc = []
                batch_f1 = []
                batch_val_f1 = []
                # last_it = [x for x,_ in enumerate(range(0, X_train.size(0), self.batch_size))][-2]
                
                for iteration, (batch_x, batch_y) in enumerate(dataloader_train):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    if (epoch == 0) and iteration == 0:
                        for c in tc.unique(y_train):
                            print(f"Proportion Class {c}: {batch_y[batch_y==c].shape[0]/len(batch_y)}")
                            
                    outputs = clf(batch_x)
                    # obtain the loss 
                    loss = criterion(outputs.flatten(), batch_y.view(-1).flatten())
                    hist[epoch] = loss.item()
                    
                    if self.task == "binary":
                        if "cuda" in self.device:
                            temp_outpouts = (outputs.cpu().detach().numpy() > self.threshold).astype(int)
                            y_batch = batch_y.view(-1).cpu().detach().numpy()
                            b_acc[epoch] = balanced_accuracy_score(y_batch,
                                                               temp_outpouts)
                        else:
                            temp_outpouts = (outputs.detach().numpy() > self.threshold).astype(int)
                            y_batch = batch_y.view(-1).detach().numpy()
                            b_acc[epoch] = balanced_accuracy_score(y_batch,
                                                           temp_outpouts)
                        batch_acc.append(b_acc[epoch])
                        batch_f1.append(f1_score(y_batch, temp_outpouts, average='binary'))
                    
                    
                    # Backprop and perform Adam optimisation
                    loss.backward()
                    optimizer.step()
                    
                    
                    if X_val is not None:
                        with tc.no_grad():
                            mini_loss = [] 
                            for batch_X_val, batch_y_val in dataloader_val:
                                batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                                self.valYhat = clf(batch_X_val)
                                val_loss = criterion(self.valYhat, batch_y_val.flatten())
                                mini_loss.append(val_loss.item())

                                if self.task == "binary":
                                    if "cuda" in self.device:
                                        temp_out_y = ( self.valYhat.cpu().detach().numpy() > self.threshold).astype(int)
                                        y_val_batch = batch_y_val.view(-1).cpu().detach().numpy()
                                        val_b_acc[epoch] =balanced_accuracy_score( y_val_batch ,
                                                                temp_out_y)
                                    else:
                                        temp_out_y = ( self.valYhat.detach().numpy() > self.threshold).astype(int)
                                        y_val_batch = batch_y_val.view(-1).detach().numpy()
                                        val_b_acc[epoch] =balanced_accuracy_score( y_val_batch,
                                                                    temp_out_y)
                                    batch_val_acc.append(val_b_acc[epoch])
                                    batch_val_f1.append(f1_score( y_val_batch, temp_out_y, average='binary'))
                            val_hist[epoch] = np.mean(mini_loss)
                    if verbose == 1:
                        if self.task == "binary":
                            if epoch % 10 == 0:
                                print("\nEpoch: %d, loss: %1.5f" % (epoch, loss.item()))
                                print("Epoch: %d, b_acc: %1.5f" % (epoch, b_acc[epoch]))
                                print("Epoch: %d, f1 (binary): %1.5f" % (epoch, f1_score( y_batch, temp_outpouts, average='binary')))
                                if X_val is not None:
                                    print("Epoch: %d, val_loss: %1.5f" % (epoch, val_hist[epoch]))
                                    print("Epoch: %d, val_b_acc: %1.5f" % (epoch, val_b_acc[epoch]))
                                    print("Epoch: %d, val_f1 (binary): %1.5f\n" % (epoch, f1_score( y_val_batch, temp_out_y, average='binary')))
                        else:
                            if epoch % print_every_epochs == 0:
                                print("\nEpoch: %d, loss: %1.5f" % (epoch, loss.item()))
                if self.task == "binary":
                    b_acc[epoch] = np.mean(batch_acc)
                    val_b_acc[epoch] = np.mean(batch_val_acc)
                    f1[epoch] = np.mean(batch_f1)
                    val_f1[epoch] = np.mean(batch_val_f1)
        except KeyboardInterrupt:
            if model_out_cpu and self.device != "cpu":
                # Puts back model on cpu 
                clf.cpu()
                clf.device = "cpu"

            # Eval/testing mode
            clf.eval()
        
            self.x_train = X_train
            self.x_test = X_val
            return clf, hist, val_hist 
        except:
            raise 
            
        if model_out_cpu and self.device != "cpu":
            # Puts back model on cpu 
            clf.cpu()
            clf.device = "cpu"

        # Eval/testing mode
        clf.eval()
        
        self.x_train = X_train
        self.x_test = X_val

        # if "cuda" in self.device:
        #     if X_val is not None:
        #         y_hat_val = clf(X_val).cpu().detach().numpy()
        #         y_val = y_val.cpu().detach().numpy()
        #     # Ensuring last self.proba correspond to training set
        #     y_hat = clf(X_train).cpu().detach().numpy()
        #     y_train = y_train.cpu().detach().numpy()
        # else:
        #     if X_val is not None:
        #         y_hat_val = clf(X_val).detach().numpy()
        #         y_val = y_val.detach().numpy()
        #     # Ensuring last self.proba correspond to training set
        #     y_hat = clf(X_train).detach().numpy()
        #     y_train = y_train.detach().numpy()
            
        # y_hat = (y_hat > self.threshold).astype(int)
        # self.trainYhat = y_hat
        # self.CM = confusion_matrix(y_train, y_hat)
        # if self.task == "binary":
        #     self.balanced_acc = balanced_accuracy_score(y_train, y_hat)
        #     self.f1 = f1
        #     self.b_acc = b_acc
        
        # if X_val is not None:
        #     y_hat_val = (y_hat_val > self.threshold).astype(int)
        #     self.valYhat = y_hat_val 
        #     self.CM_val = confusion_matrix(y_val, y_hat_val)
        #     if self.task == "binary":
        #         self.val_b_acc = val_b_acc
        #         self.f1_val = val_f1
                
            
        return clf, hist, val_hist

    def train_Precursor_mc(self, clf, X_train, y_train,
                           X_val=None, y_val=None, l2=0,
                           num_epochs=200, learning_rate=0.01, verbose=0,
                           model_out_cpu=True, **kw):

        # Convert to parameters CUDA Tensors
        try:
            clf = clf.to(self.device)
        except:
            pass
        try:
            clf.train()
        except:
            pass

        self.n_epochs = num_epochs
        print_every_epochs = kw.pop("print_every_epochs", 10)
        average = kw.pop("average", "weighted")
        self.optimizer = kw.pop("optimizer", "adam")
        hard_th = kw.pop("hard_th", False)
        # Binary cross entropy loss, learning rate and l2 regularization
        weight = kw.pop("class_weight", None)
        if weight is not None:
            weight = tc.Tensor(weight).to(self.device)

        criterion = tc.nn.BCEWithLogitsLoss(weight=weight)
        self.task = "multiclass"
        n_class = len(np.unique(y_train))

        print("Classification task: {}".format(self.task))
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam(clf.parameters(),
                                      lr=learning_rate, weight_decay=l2)
        elif self.optimizer == "SGD":
            optimizer = tc.optim.SGD(clf.parameters(), momentum=0.999,
                                     lr=learning_rate, weight_decay=l2)

        # Init loss history and balanced accuracy
        hist = np.zeros(num_epochs)
        val_hist = np.zeros(num_epochs)
        b_acc = np.zeros(num_epochs)
        val_b_acc = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)
        val_f1 = np.zeros(num_epochs)

        # Conversion to tensors
        if not tc.is_tensor(X_train):
            X_train = tc.Tensor(X_train)
        if not tc.is_tensor(y_train):
            y_train = tc.Tensor(y_train.flatten())
            y_train = y_train.type(tc.int64)

        if X_val is not None:
            if not tc.is_tensor(X_val):
                X_val = tc.Tensor(X_val)
            if not tc.is_tensor(y_val):
                y_val = tc.Tensor(y_val)
                y_val = y_val.type(tc.int64)
            data_val = myDataset(X_val, y_val)

        if self.batch_size is None:
            self.batch_size = X_train.size(0)
            warnings.warn("Setting the batch size = full training set could overwhelm GPU memory")

        data_train = myDataset(X_train, y_train)
        if self.use_stratified_batch_size is False:
            print("Mini-batch strategy: Random sampling")
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        else:
            print("Mini-batch strategy: Stratified")
            # get class counts
            # class_sample_counts = []
            weights = []
            for label in tc.unique(y_train):
                count = len(tc.where(y_train == label)[0])
                # class_sample_counts.append(count)
                weights.append(1/count)
            # print(1/tc.tensor(class_sample_counts))
            weights = tc.tensor(weights).to(self.device)
            samples_weights = weights[y_train.type(tc.LongTensor).to(self.device)]
            sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, sampler=sampler)
        if X_val is not None:
            dataloader_val = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

        # Train the model
        try:
            for epoch in tqdm(range(num_epochs)):
                batch_acc = []
                batch_val_acc = []
                batch_f1 = []
                batch_val_f1 = []
                for iteration, (batch_x, batch_y) in enumerate(dataloader_train):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    if (epoch == 0) and iteration == 0:
                        for c in tc.unique(y_train):
                            print(f"Proportion Class {c}: {batch_y[batch_y == c].shape[0] / len(batch_y)}")

                    outputs = clf(batch_x)
                    # One-hot encoding
                    # y_temp = tc.zeros((len(batch_y), n_class))
                    # y = tc.tensor(batch_y).type(tc.int64).to(self.device)
                    # y_temp.scatter_(1, y, 1)
                    y_temp = one_hot_embedding(batch_y.flatten(), n_class)
                    # obtain the loss
                    if hard_th:
                        bool_mat = nn.Sigmoid()(outputs) <= self.threshold
                        outputs[bool_mat] = -10 # large logits
                    loss = criterion(outputs, y_temp.to(self.device))
                    hist[epoch] = loss.item()

                    if "cuda" in self.device:
                        temp_outpouts = tc.argmax(outputs,
                                                  dim=-1).cpu().detach().numpy()  # > self.threshold).astype(int)
                        batch_y_cpu = batch_y.cpu().detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(batch_y_cpu,
                                                               temp_outpouts)

                    else:
                        temp_outpouts = tc.argmax(outputs, dim=-1).detach().numpy()
                        batch_y_cpu = batch_y.detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(batch_y_cpu,
                                                               temp_outpouts)
                    batch_acc.append(b_acc[epoch])
                    batch_f1.append(f1_score(batch_y_cpu,
                                             temp_outpouts, average=average))
                    # Backprop and perform Adam optimisation
                    loss.backward()
                    optimizer.step()

                    if X_val is not None:
                        with tc.no_grad():
                            mini_loss = []
                            for batch_X_val, batch_y_val in dataloader_val:
                                self.valYhat = clf(batch_X_val)

                                # One-hot encoding
                                # y_temp_val = tc.zeros((len(batch_y_val), n_class))
                                # y = tc.tensor(batch_y_val).type(tc.int64).to(self.device)
                                # y_temp_val.scatter_(1, y, 1)
                                y_temp_val = one_hot_embedding(batch_y_val.flatten(), n_class)
                                val_loss = criterion(self.valYhat, y_temp_val.to(self.device))
                                mini_loss.append(val_loss.item())

                                if "cuda" in self.device:
                                    temp_out_y = tc.argmax(self.valYhat, dim=-1).cpu().detach().numpy()
                                    y_val_batch = batch_y_val.cpu().detach().numpy()
                                    val_b_acc[epoch] = balanced_accuracy_score(y_val_batch,
                                                                               temp_out_y)
                                else:
                                    temp_out_y = tc.argmax(self.valYhat, dim=-1).detach().numpy()
                                    y_val_batch = batch_y_val.detach().numpy()
                                    val_b_acc[epoch] = balanced_accuracy_score(y_val_batch,
                                                                               temp_out_y)
                                batch_val_acc.append(val_b_acc[epoch])
                                batch_val_f1.append(f1_score(y_val_batch, temp_out_y, average=average))
                            val_hist[epoch] = np.mean(mini_loss)
                    if verbose == 1:
                        if epoch % print_every_epochs == 0:
                            print("\nEpoch: %d, loss: %1.5f" % (epoch, loss.item()))
                            print("Epoch: %d, b_acc: %1.5f" % (epoch, b_acc[epoch]))
                            print("Epoch: %d, f1 (%s): %1.5f" % (
                            epoch, average, np.mean(batch_f1)))
                            if X_val is not None:
                                print("Epoch: %d, val_loss: %1.5f" % (epoch, val_hist[epoch]))
                                print("Epoch: %d, val_b_acc: %1.5f" % (epoch, val_b_acc[epoch]))
                                print("Epoch: %d, val_f1 (%s): %1.5f\n" % (
                                epoch, average, np.mean(np.mean(batch_val_f1))))

                b_acc[epoch] = np.mean(batch_acc)
                val_b_acc[epoch] = np.mean(batch_val_acc)
                f1[epoch] = np.mean(batch_f1)
                val_f1[epoch] = np.mean(batch_val_f1)
        except KeyboardInterrupt:
            if model_out_cpu and self.device != "cpu":
                # Puts back model on cpu
                clf.cpu()
                clf.device = "cpu"

            # Eval/testing mode
            clf.eval()

            self.x_train = X_train
            self.x_test = X_val
            return clf, hist, val_hist
        except:
            raise

        if model_out_cpu and self.device != "cpu":
            # Puts back model on cpu
            clf.cpu()
            clf.device = "cpu"

        # Eval/testing mode
        clf.eval()

        self.x_train = X_train
        self.x_test = X_val

        return clf, hist, val_hist


    def evaluate_model(self, mode="train", use_cuda=False,
		batch_size=None, get_important_features=False):
      if batch_size == None:
        batch_size = self.batch_size
      if use_cuda:
        try:
          self.trainedModel.cuda()
          self.trainedModel.device = "cuda:0"
        except:
          pass
      if mode == "train":
        y_hat = tc.zeros(len(self.x_train))
        top_features, top_features_values= [], []
        time_proba = np.zeros((self.x_train.shape[0], self.Wn))
        for iteration, i in enumerate(range(0, self.x_train.shape[0], batch_size)):
          if self.n_classes ==1:
            temp_y =  self.trainedModel.predict(self.x_train[i:i+batch_size], train=False)
            temp_proba = self.trainedModel.proba_time.squeeze().detach().numpy()
          else:
            temp_y = self.trainedModel.predict(self.x_train[i:i+batch_size], train=False)[1]
            N = len(self.x_train[i:i+batch_size])
            temp_proba = self.trainedModel.proba_time[np.arange(N), :, temp_y].squeeze().detach().numpy()
          y_hat[i:i+batch_size] = temp_y
          time_proba[i:i+batch_size] = temp_proba
          if get_important_features:
            for e, flight_id in enumerate(range(0, batch_size)):
              try:
                self.trainedModel.show_feature_importance(self.header, flight_id, plot=False, class_interest=temp_y[e])
                if self.trainedModel.multiple_precursors:
                    top_features.append(self.trainedModel.sorted_features)
                    top_features_values.append(self.trainedModel.sorted_features_values)
                else:
                    top_features.append(self.trainedModel.list_sorted_features[1])
                    top_features_values.append(self.trainedModel.list_sorted_features_values[1])
              except:
                top_features.append([None])
                top_features_values.append([None])
        if get_important_features:
          return y_hat,time_proba, top_features, top_features_values
        else:
          return y_hat, time_proba
      elif mode == "val":
        y_hat = tc.zeros(len(self.x_test))
        top_features, top_features_values = [], []
        time_proba = np.zeros((self.x_test.shape[0], self.Wn))
        for iteration, i in enumerate(range(0, self.x_test.shape[0], batch_size)):
          if self.n_classes == 1:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)
              temp_proba = self.trainedModel.proba_time.squeeze().detach().numpy()
          else:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)[1]
              N = len(self.x_test[i:i+batch_size])
              temp_proba = self.trainedModel.proba_time[np.arange(N), :, temp_y].squeeze().detach().numpy()
          y_hat[i:i+batch_size] = temp_y
          time_proba[i:i+batch_size] = temp_proba
          if get_important_features:
            limit = min([batch_size, len(self.x_test)])
            for e, flight_id in enumerate(range(0, limit)):
              try:
                  self.trainedModel.show_feature_importance(self.header, flight_id, plot=False, class_interest=temp_y[e])
                  if self.trainedModel.multiple_precursors:
                      top_features.append(self.trainedModel.sorted_features)
                      top_features_values.append(self.trainedModel.sorted_features_values)
                  else:
                      top_features.append(self.trainedModel.list_sorted_features[1])
                      top_features_values.append(self.trainedModel.list_sorted_features_values[1])
              except:
                top_features.append([None])
                top_features_values.append([None])
        if get_important_features:
          return y_hat,time_proba, top_features, top_features_values
        else:
          return y_hat, time_proba
      else:
        y_hat_train = tc.zeros(len(self.x_train))
        y_hat_val = tc.zeros(len(self.x_test))
        for iteration, i in enumerate(range(0, self.x_train.size(0), batch_size)):
          if self.n_classes == 1:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)
          else:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)[1]
          y_hat_train[i:i+batch_size] = temp_y
        for iteration, i in enumerate(range(0, self.x_test.size(0), self.batch_size)):
          if self.n_classes == 1:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)
          else:
              temp_y = self.trainedModel.predict(self.x_test[i:i+batch_size], train=False)[1]
          y_hat_val[i:i+batch_size] = temp_y
        return y_hat_train, y_hat_val

### Binary Models

class MILLR(tc.nn.Module):
    def __init__(self, input_dim, flight_length, device, aggregation="maxpool"):
        super(MILLR, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.flight_length = flight_length
        self.D = input_dim
        self.device = device
        self.task = "binary"
        self.threshold = 0.5
        self.agg = aggregation

    def forward(self, x, train=True):
        N, _, _ = x.size()
        self.pi = self.sigmoid(self.fc(x)).squeeze()  # NxL
        self.proba_time = self.pi
        if self.agg == "mean":
            p = tc.mean(self.pi, axis=-1)  # Nx1
        elif self.agg == "maxpool":
            p = tc.max(self.pi, dim=-1)[0]
        return p

    def get_feature_importance(self, columns, n_top=5):
        coeffs = self.fc.weight.flatten().detach().numpy()
        sorted_feat_idx = np.argsort(coeffs)[::-1]
        sorted_columns = columns[sorted_feat_idx[:n_top]]
        top_values = coeffs[sorted_feat_idx[:n_top]]
        return sorted_columns, top_values

    def cross_time_steps_loss(self, Pi):
        diff = (Pi[:, :-1] - Pi[:, 1:]) ** 2
        return tc.mean(tc.mean(diff, axis=-1))

    def train_LR(self, X_train, y_train, X_val, y_val, batch_size, print_every_epochs=5,
                 l2=0.001, learning_rate=0.001, use_stratified_batch_size=True, verbose=1,
                 num_epochs=100, optimizer="adam", momentum=0.99):
        self.train()
        if "cuda" in self.device:
            self.cuda()
        else:
            self.cpu()

        self.batch_size = batch_size
        criterion = nn.BCELoss()
        if optimizer == "adam":
            optimizer = tc.optim.Adam(self.parameters(),
                                      lr=learning_rate, weight_decay=l2)
        else:
            optimizer = tc.optim.SGD(self.parameters(), momentum=momentum,
                                     lr=learning_rate, weight_decay=l2)
        hist = np.zeros(num_epochs)
        val_hist = np.zeros(num_epochs)
        b_acc = np.zeros(num_epochs)
        val_b_acc = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)
        val_f1 = np.zeros(num_epochs)

        # Conversion to tensors
        if not tc.is_tensor(X_train):
            X_train = tc.Tensor(X_train)
        if not tc.is_tensor(y_train):
            y_train = tc.Tensor(y_train.flatten())

        if X_val is not None:
            if not tc.is_tensor(X_val):
                X_val = tc.Tensor(X_val)
            if not tc.is_tensor(y_val):
                y_val = tc.Tensor(y_val)
            data_val = myDataset(X_val, y_val)

        data_train = myDataset(X_train, y_train)
        if use_stratified_batch_size is False:
            print("Mini-batch strategy: Random sampling")
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        else:
            print("Mini-batch strategy: Stratified")
            # get class counts
            weights = []
            for label in tc.unique(y_train):
                count = len(tc.where(y_train == label)[0])
                weights.append(1 / count)
            weights = tc.tensor(weights).to(self.device)
            samples_weights = weights[y_train.type(tc.LongTensor).to(self.device)]
            sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
            dataloader_train = DataLoader(data_train, batch_size=self.batch_size, sampler=sampler)
        if X_val is not None:
            dataloader_val = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

        try:
            for epoch in tqdm(range(num_epochs)):
                batch_acc = []
                batch_val_acc = []
                batch_f1 = []
                batch_val_f1 = []

                for iteration, (batch_x, batch_y) in enumerate(dataloader_train):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    if (epoch == 0) and iteration == 0:
                        for c in tc.unique(y_train):
                            print(f"Proportion Class {c}: {batch_y[batch_y == c].shape[0] / len(batch_y)}")

                    outputs = self.forward(batch_x)
                    g_loss = self.cross_time_steps_loss(self.pi)
                    # obtain the loss
                    loss = criterion(outputs.flatten(), batch_y.view(-1).flatten()) + g_loss
                    hist[epoch] = loss.item()

                    if "cuda" in self.device:
                        temp_outpouts = (outputs.cpu().detach().numpy() > self.threshold).astype(int)
                        y_batch = batch_y.view(-1).cpu().detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(y_batch,
                                                               temp_outpouts)
                    else:
                        temp_outpouts = (outputs.detach().numpy() > self.threshold).astype(int)
                        y_batch = batch_y.view(-1).detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(y_batch,
                                                               temp_outpouts)
                    batch_acc.append(b_acc[epoch])
                    batch_f1.append(f1_score(y_batch, temp_outpouts, average='binary'))

                    # Backprop and perform Adam optimisation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if X_val is not None:
                        with tc.no_grad():
                            mini_loss = []
                            for batch_X_val, batch_y_val in dataloader_val:
                                batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                                self.valYhat = self.forward(batch_X_val)
                                g_loss_val = self.cross_time_steps_loss(self.pi)
                                val_loss = criterion(self.valYhat, batch_y_val.flatten()) + g_loss_val
                                mini_loss.append(val_loss.item())

                                if self.task == "binary":
                                    if "cuda" in self.device:
                                        temp_out_y = (self.valYhat.cpu().detach().numpy() > self.threshold).astype(int)
                                        y_val_batch = batch_y_val.view(-1).cpu().detach().numpy()
                                        val_b_acc[epoch] = balanced_accuracy_score(y_val_batch,
                                                                                   temp_out_y)
                                    else:
                                        temp_out_y = (self.valYhat.detach().numpy() > self.threshold).astype(int)
                                        y_val_batch = batch_y_val.view(-1).detach().numpy()
                                        val_b_acc[epoch] = balanced_accuracy_score(y_val_batch,
                                                                                   temp_out_y)
                                    batch_val_acc.append(val_b_acc[epoch])
                                    batch_val_f1.append(f1_score(y_val_batch, temp_out_y, average='binary'))
                            val_hist[epoch] = np.mean(mini_loss)
                    if verbose == 1:
                        if self.task == "binary":
                            if epoch % 10 == 0:
                                print("\nEpoch: %d, loss: %1.5f" % (epoch, loss.item()))
                                print("Epoch: %d, b_acc: %1.5f" % (epoch, b_acc[epoch]))
                                print("Epoch: %d, f1 (binary): %1.5f" % (
                                epoch, f1_score(y_batch, temp_outpouts, average='binary')))
                                if X_val is not None:
                                    print("Epoch: %d, val_loss: %1.5f" % (epoch, val_hist[epoch]))
                                    print("Epoch: %d, val_b_acc: %1.5f" % (epoch, val_b_acc[epoch]))
                                    print("Epoch: %d, val_f1 (binary): %1.5f\n" % (
                                    epoch, f1_score(y_val_batch, temp_out_y, average='binary')))
                        else:
                            if epoch % print_every_epochs == 0:
                                print("\nEpoch: %d, loss: %1.5f" % (epoch, loss.item()))
                if self.task == "binary":
                    b_acc[epoch] = np.mean(batch_acc)
                    val_b_acc[epoch] = np.mean(batch_val_acc)
                    f1[epoch] = np.mean(batch_f1)
                    val_f1[epoch] = np.mean(batch_val_f1)
        except KeyboardInterrupt:
            self.cpu()
            self.device = "cpu"
            # Eval/testing mode
            self.eval()
            self.x_train = X_train
            self.x_test = X_val
            self.hist = hist
            self.val_hist = val_hist
        except:
            raise

        self.cpu()
        self.device = "cpu"
        self.eval()
        self.x_train = X_train
        self.x_test = X_val
        self.hist = hist
        self.val_hist = val_hist

    def fit(self, **kw):
        self.train_LR(**kw)

class adopt_like(nn.Module, ModelContainer): 
  def __init__(self, input_size, num_flights, flight_len,
                window_step, sequence_length=20,
                n_filters_per_feature=1, hidden_size=5,
                **kwargs):
        ModelContainer.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.model_architecture = "ADOPT"

        self.input_size = input_size
        self.window_step = window_step
        self.sequence_length = sequence_length
        self.num_flights = num_flights
        self.n_windows = int((flight_len-sequence_length)/window_step +1)
  
        self.time_step_proba = tc.zeros(num_flights, flight_len)
        self.flight_len = flight_len
        self.n_filters_per_feature = n_filters_per_feature

      
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size= self.input_size, 
                                        hidden_size = self.hidden_size, batch_first=True)
        self.tanh_ = nn.Tanh()
        self.dense1 = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                              out_features= 500), 
                                    nn.Tanh())
        self.dense2 = nn.Sequential(nn.Linear(in_features=500,
                                              out_features= 1), 
                                    nn.Sigmoid())
         
  def forward(self, x):
      if not tc.is_tensor(x):
              x = tc.Tensor(x).to(self.device)
      else:
              x = x.to(self.device)
    
      self.pooled_proba_features = tc.zeros((x.size(0), x.size(2))).to(self.device)
      self.precursor_proba = tc.zeros((x.size(0), 
                                        self.n_windows,
                                        x.size(2))).to(self.device)

      self.proba_time = tc.zeros((x.size(0), self.n_windows)).to(self.device)  

      h_0 = tc.zeros(1, x.size(0), self.hidden_size).to(self.device)
      gru_out, h_out = self.gru(x, h_0)
      out1 = self.dense1(self.tanh_(gru_out))
      self.proba_time = self.dense2(out1) 
      final_out  = tc.max(self.proba_time, axis=1)[0]


      return final_out.view(-1)
  
  def fit(self, threshold=0.5, batch_size=None, **kwargs):
      self.threshold = threshold
      self.batch_size = batch_size
      self.trainedModel, self.hist, self.val_hist = self.train_Precursor_binary(**kwargs)

  def predict(self, x):
    with tc.no_grad():
        out = self.trainedModel(x)
    return out

class precursor_model(nn.Module, ModelContainer):
    def __init__(self, input_size, sequence_length,
                 kernel_size, n_filters,
                 fourth_layer, hidden_size=5, n_classes=1,
                 **kwargs):
        ModelContainer.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.model_architecture = "CNN + CNN + CNN + CNN + GRU + Dense"

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.sequence_length = sequence_length
        self.fourth_layer = fourth_layer
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.compute_window_size()

        # Multi-headed
        self.cnns1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters[0],
                      kernel_size=kernel_size[0],
                      stride=1),
            # nn.BatchNorm1d(16),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[0], out_channels=n_filters[1],
                      kernel_size=kernel_size[1],
                      stride=1),
            # nn.BatchNorm1d(32),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[1], out_channels=n_filters[2],
                      kernel_size=kernel_size[2],
                      stride=1),
            # nn.BatchNorm1d(64),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        if fourth_layer == "cnn":
            self.cnns4 = nn.ModuleList([nn.Sequential(
                nn.Conv1d(in_channels=n_filters[2], out_channels=1,
                          kernel_size=1, stride=1),
                # nn.BatchNorm1d(1),
                #             nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        elif fourth_layer == "dense":
            self.dense = nn.ModuleList([nn.Sequential(
                        nn.Linear(in_features=n_filters[2], out_features=1),
            #           nn.Dropout(0.1),
                        nn.Sigmoid()) for i in range(self.input_size)])
        else:
            raise Exception("Fourth layer can only be dense or cnn")

        # Common
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          batch_first=True)
        self.tanh = nn.Tanh()

        self.Dense_final = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                      out_features=n_classes),
            nn.Sigmoid())

    def forward(self, x):
        if not tc.is_tensor(x):
            x = tc.Tensor(x).to(self.device)
        else:
            x = x.to(self.device)

        self.precursor_proba = tc.zeros((x.size(0),
                                         self.Wn,
                                         x.size(2))).to(self.device)

        # for each feature
        for dim in range(x.shape[-1]):
            fl = x[:, :, dim].unsqueeze(-1)  # create additional dimension
            fl = fl.permute(0, 2, 1)  # Swap axis (dimension and time steps) for cnn input [batch_size, 1, time-steps]
            out1 = self.cnns1[dim](fl)  # [n_flight, dim=n_filters_per_feature, window]
            out2 = self.cnns2[dim](out1)  # [n_flight, dim=1, window]
            out3 = self.cnns3[dim](out2)
            if self.fourth_layer == "cnn":
                out4 = self.cnns4[dim](out3)
                temp = out4.permute(0, 2, 1)
                self.precursor_proba[:, :, dim] = \
                    temp.view(x.size(0), -1)  # [n_flight, window], dim=1
            elif self.fourth_layer == "dense":
                out4 = self.dense[dim](out3.permute(0, 2, 1))
                self.precursor_proba[:, :, dim] = \
                    out4.view(x.size(0), -1)  # [n_flight, window], dim=1



        h_0 = tc.zeros(1, x.size(0), self.hidden_size).to(self.device)
        gru_out, _ = self.gru(self.precursor_proba, h_0)
        self.proba_time = self.Dense_final(self.tanh(gru_out))
        final_out = tc.max(self.proba_time, axis=1)[0]  # Max across time

        return final_out.view(-1)

    def fit(self, threshold=0.5, batch_size=None, **kwargs):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_stratified_batch_size = kwargs.pop("use_stratified_batch_size", False)
        self.trainedModel, self.hist, self.val_hist = self.train_Precursor_binary(**kwargs)

    def predict(self, x):
        with tc.no_grad():
            out = self.trainedModel(x)
        return out

    def compute_window_size(self):
        # Assuming stride = 1 and 3 CNN
        self.Wn = (((self.sequence_length-self.kernel_size[0]+1) - self.kernel_size[1]+1)- self.kernel_size[2]+1)

class precursor_model_BN(nn.Module, ModelContainer):
    def __init__(self, input_size, sequence_length,
                 kernel_size, n_filters,
                 fourth_layer, hidden_size=5, n_classes=1,
                 **kwargs):
        ModelContainer.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.model_architecture = "CNN + CNN + CNN + CNN + GRU + Dense"

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.sequence_length = sequence_length
        self.fourth_layer = fourth_layer
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.compute_window_size()

        # Multi-headed
        self.cnns1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters[0],
                      kernel_size=kernel_size[0],
                      stride=1),
            nn.BatchNorm1d(n_filters[0]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[0], out_channels=n_filters[1],
                      kernel_size=kernel_size[1],
                      stride=1),
            nn.BatchNorm1d(n_filters[1]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[1], out_channels=n_filters[2],
                      kernel_size=kernel_size[2],
                      stride=1),
            nn.BatchNorm1d(n_filters[2]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        if fourth_layer == "cnn":
            self.cnns4 = nn.ModuleList([nn.Sequential(
                nn.Conv1d(in_channels=n_filters[2], out_channels=1,
                          kernel_size=1, stride=1),
                #             nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        elif fourth_layer == "dense":
            self.dense = nn.ModuleList([nn.Sequential(
                nn.Linear(in_features=n_filters[2], out_features=1),
                #           nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        else:
            raise Exception("Fourth layer can only be dense or cnn")

        # Common
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          batch_first=True)
        self.tanh = nn.Tanh()
        
        if self.n_classes == 1:
            self.Dense_final = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.n_classes),
            nn.Sigmoid())
        else:
            self.Dense_final = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.n_classes)
            )

    def forward(self, x, train=True):
        if not tc.is_tensor(x):
            x = tc.Tensor(x).to(self.device)
        else:
            x = x.to(self.device)

        self.precursor_proba = tc.zeros((x.size(0),
                                         self.Wn,
                                         x.size(2))).to(self.device)

        self.convolution_ouputs = []
        # for each feature
        for dim in range(x.shape[-1]):
            conv_outputs_feature= []
            fl = x[:, :, dim].unsqueeze(-1)  # create additional dimension
            fl = fl.permute(0, 2, 1)  # Swap axis (dimension and time steps) for cnn input [batch_size, 1, time-steps]
            out1 = self.cnns1[dim](fl)  # [n_flight, dim=n_filters_per_feature, window]
            conv_outputs_feature.append(out1.permute(0, 2, 1))
            out2 = self.cnns2[dim](out1)  # [n_flight, dim=1, window]
            conv_outputs_feature.append(out2.permute(0, 2, 1))
            out3 = self.cnns3[dim](out2)
            conv_outputs_feature.append(out3.permute(0, 2, 1))
            if self.fourth_layer == "cnn":
                out4 = self.cnns4[dim](out3)
                temp = out4.permute(0, 2, 1)
                self.precursor_proba[:, :, dim] = \
                    temp.view(x.size(0), -1)  # [n_flight, window], dim=1
            elif self.fourth_layer == "dense":
                out4 = self.dense[dim](out3.permute(0, 2, 1))
                self.precursor_proba[:, :, dim] = \
                    out4.view(x.size(0), -1)  # [n_flight, window], dim=1
            self.convolution_ouputs.append(conv_outputs_feature)
        # # Set non influential signals to zero
        # self.precursor_proba[(self.precursor_proba>=0.49) & (self.precursor_proba<=0.51)] = 0
        h_0 = tc.zeros(1, x.size(0), self.hidden_size).to(self.device)
        gru_out, _ = self.gru(self.precursor_proba, h_0)
        self.proba_time = self.Dense_final(self.tanh(gru_out))
        final_out = tc.max(self.proba_time, axis=1)[0]  # Max across time
        
        if self.task == "binary":
            return final_out.view(-1)
        else:
            return final_out

    def fit(self, threshold=0.5, batch_size=None, use_stratified_batch_size=False, **kwargs):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_stratified_batch_size = use_stratified_batch_size
        self.trainedModel, self.hist, self.val_hist = self.train_Precursor_binary(**kwargs)

    def predict(self, x, train=False):
        with tc.no_grad():
            if self.task == "binary":
                if (self.device != "cpu") and (not next(self.parameters()).is_cuda):
                    try:
                        self.trainedModel.cuda(self.device)
                    except:
                        pass
                elif (self.device == "cpu") and (next(self.parameters()).is_cuda):
                    try:
                        self.trainedModel.cpu() 
                    except:
                        pass
                     
                out = self.trainedModel(x, train)
            else:
                out = F.softmax(self.trainedModel(x, train), dim=1)
        return out

    def compute_window_size(self):
        # Assuming stride = 1 and 3 CNN layers
        self.Wn = (((self.sequence_length - self.kernel_size[0] + 1) - self.kernel_size[1] + 1) - self.kernel_size[2] + 1)


class precursor_model_BN_v2(nn.Module, ModelContainer):
    def __init__(self, input_size, sequence_length,
                 kernel_size, n_filters,
                 fourth_layer, hidden_size=5, n_classes=2,
                 **kwargs):
        ModelContainer.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.model_architecture = "CNN + CNN + CNN + CNN + GRU + Dense"

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.sequence_length = sequence_length
        self.fourth_layer = fourth_layer
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.compute_window_size()

        # Multi-headed
        self.cnns1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters[0],
                      kernel_size=kernel_size[0],
                      stride=1),
            nn.BatchNorm1d(n_filters[0]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[0], out_channels=n_filters[1],
                      kernel_size=kernel_size[1],
                      stride=1),
            nn.BatchNorm1d(n_filters[1]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[1], out_channels=n_filters[2],
                      kernel_size=kernel_size[2],
                      stride=1),
            nn.BatchNorm1d(n_filters[2]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        if fourth_layer == "cnn":
            self.cnns4 = nn.ModuleList([nn.Sequential(
                nn.Conv1d(in_channels=n_filters[2], out_channels=1,
                          kernel_size=1, stride=1),
                #             nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        elif fourth_layer == "dense":
            self.dense = nn.ModuleList([nn.Sequential(
                nn.Linear(in_features=n_filters[2], out_features=1),
                #           nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        else:
            raise Exception("Fourth layer can only be dense or cnn")

        # Common
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          batch_first=True)
        self.tanh = nn.Tanh()

        if self.n_classes == 1:
            self.Dense_final = nn.Sequential(
                nn.Linear(in_features=self.hidden_size,
                          out_features=self.n_classes),
                nn.Sigmoid())
        else:
            self.Dense_final = nn.Sequential(
                nn.Linear(in_features=self.hidden_size,
                          out_features=self.n_classes)
            )

    def forward(self, x):
        if not tc.is_tensor(x):
            x = tc.Tensor(x).to(self.device)
        else:
            x = x.to(self.device)

        self.precursor_proba = tc.zeros((x.size(0),
                                         self.Wn,
                                         x.size(2))).to(self.device)

        self.convolution_ouputs = []
        # for each feature
        for dim in range(x.shape[-1]):
            conv_outputs_feature = []
            fl = x[:, :, dim].unsqueeze(-1)  # create additional dimension
            fl = fl.permute(0, 2, 1)  # Swap axis (dimension and time steps) for cnn input [batch_size, 1, time-steps]
            out1 = self.cnns1[dim](fl)  # [n_flight, dim=n_filters_per_feature, window]
            conv_outputs_feature.append(out1.permute(0, 2, 1))
            out2 = self.cnns2[dim](out1)  # [n_flight, dim=1, window]
            conv_outputs_feature.append(out2.permute(0, 2, 1))
            out3 = self.cnns3[dim](out2)
            conv_outputs_feature.append(out3.permute(0, 2, 1))
            if self.fourth_layer == "cnn":
                out4 = self.cnns4[dim](out3)
                temp = out4.permute(0, 2, 1)
                self.precursor_proba[:, :, dim] = \
                    temp.view(x.size(0), -1)  # [n_flight, window], dim=1
            elif self.fourth_layer == "dense":
                out4 = self.dense[dim](out3.permute(0, 2, 1))
                self.precursor_proba[:, :, dim] = \
                    out4.view(x.size(0), -1)  # [n_flight, window], dim=1
            self.convolution_ouputs.append(conv_outputs_feature)
        # # Set non influential signals to zero
        # self.precursor_proba[(self.precursor_proba>=0.49) & (self.precursor_proba<=0.51)] = 0
        h_0 = tc.zeros(1, x.size(0), self.hidden_size).to(self.device)
        gru_out, _ = self.gru(self.precursor_proba, h_0)
        self.proba_time = self.Dense_final(self.tanh(gru_out))
        final_out = tc.max(self.proba_time, axis=1)[0]  # Max across time

        if self.task == "binary":
            return final_out.view(-1)
        else:
            return final_out

    def fit(self, threshold=0.5, batch_size=None, use_stratified_batch_size=False, **kwargs):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_stratified_batch_size = use_stratified_batch_size
        self.trainedModel, self.hist, self.val_hist = self.train_Precursor_binary(**kwargs)

    def predict(self, x):
        with tc.no_grad():
            if self.task == "binary":
                if (self.device != "cpu") and (not next(self.parameters()).is_cuda):
                    try:
                        self.trainedModel.cuda(self.device)
                    except:
                        pass
                elif (self.device == "cpu") and (next(self.parameters()).is_cuda):
                    try:
                        self.trainedModel.cpu()
                    except:
                        pass

                out = self.trainedModel(x)
            else:
                out = F.softmax(self.trainedModel(x), dim=1)
        return out

    def compute_window_size(self):
        # Assuming stride = 1 and 3 CNN layers
        self.Wn = (((self.sequence_length - self.kernel_size[0] + 1) - self.kernel_size[1] + 1) - self.kernel_size[
            2] + 1)

#### Multiclass/multilabel models

class precursor_model_BN_mc(nn.Module, ModelContainer):
    def __init__(self, input_size, sequence_length,
                 kernel_size, n_filters,
                 fourth_layer, hidden_size=5, n_classes=2,
                 **kwargs):
        ModelContainer.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.model_architecture = "CNN + CNN + CNN + CNN + GRU + Dense"

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.sequence_length = sequence_length
        self.fourth_layer = fourth_layer
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.compute_window_size()

        # Multi-headed
        self.cnns1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters[0],
                      kernel_size=kernel_size[0],
                      stride=1),
            nn.BatchNorm1d(n_filters[0]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[0], out_channels=n_filters[1],
                      kernel_size=kernel_size[1],
                      stride=1),
            nn.BatchNorm1d(n_filters[1]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        self.cnns3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=n_filters[1], out_channels=n_filters[2],
                      kernel_size=kernel_size[2],
                      stride=1),
            nn.BatchNorm1d(n_filters[2]),
            #             nn.Dropout(0.1),
            nn.ReLU()) for i in range(self.input_size)])

        if fourth_layer == "cnn":
            self.cnns4 = nn.ModuleList([nn.Sequential(
                nn.Conv1d(in_channels=n_filters[2], out_channels=1,
                          kernel_size=1, stride=1),
                #             nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        elif fourth_layer == "dense":
            self.dense = nn.ModuleList([nn.Sequential(
                nn.Linear(in_features=n_filters[2], out_features=1),
                #           nn.Dropout(0.1),
                nn.Sigmoid()) for i in range(self.input_size)])
        else:
            raise Exception("Fourth layer can only be dense or cnn")

        # Common
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          batch_first=True)
        self.tanh = nn.Tanh()

        self.Dense_final = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.n_classes)
        )
        self.pool = tc.nn.MaxPool1d(kernel_size=self.Wn)

    def forward(self, x, train=True):
        if not tc.is_tensor(x):
            x = tc.Tensor(x).to(self.device)
        else:
            x = x.to(self.device)

        self.precursor_proba = tc.zeros((x.size(0),
                                         self.Wn,
                                         x.size(2))).to(self.device)

        self.convolution_ouputs = []
        # for each feature
        for dim in range(x.shape[-1]):
            conv_outputs_feature = []
            fl = x[:, :, dim].unsqueeze(-1)  # create additional dimension
            fl = fl.permute(0, 2, 1)  # Swap axis (dimension and time steps) for cnn input [batch_size, 1, time-steps]
            out1 = self.cnns1[dim](fl)  # [n_flight, dim=n_filters_per_feature, window]
            conv_outputs_feature.append(out1.permute(0, 2, 1))
            out2 = self.cnns2[dim](out1)  # [n_flight, dim=1, window]
            conv_outputs_feature.append(out2.permute(0, 2, 1))
            out3 = self.cnns3[dim](out2)
            conv_outputs_feature.append(out3.permute(0, 2, 1))
            if self.fourth_layer == "cnn":
                out4 = self.cnns4[dim](out3)
                temp = out4.permute(0, 2, 1)
                self.precursor_proba[:, :, dim] = \
                    temp.view(x.size(0), -1)  # [n_flight, window], dim=1
            elif self.fourth_layer == "dense":
                out4 = self.dense[dim](out3.permute(0, 2, 1))
                self.precursor_proba[:, :, dim] = \
                    out4.view(x.size(0), -1)  # [n_flight, window], dim=1
            self.convolution_ouputs.append(conv_outputs_feature)
        # # Set non influential signals to zero
        # self.precursor_proba[(self.precursor_proba>=0.49) & (self.precursor_proba<=0.51)] = 0
        h_0 = tc.zeros(1, x.size(0), self.hidden_size).to(self.device)
        gru_out, _ = self.gru(self.precursor_proba, h_0)
        self.proba_time = self.Dense_final(self.tanh(gru_out))
        final_out = self.pool(self.proba_time.permute(0,2,1)).squeeze()
        if not train:
            self.proba_time = nn.Sigmoid()(self.proba_time)

        return final_out

    def fit(self, threshold=0.5, batch_size=None, use_stratified_batch_size=False, **kwargs):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_stratified_batch_size = use_stratified_batch_size
        self.trainedModel, self.hist, self.val_hist = self.train_Precursor_mc(**kwargs)

    def predict(self, x, train=False):
        with tc.no_grad():
            if (self.device != "cpu") and (not next(self.parameters()).is_cuda):
                try:
                    self.trainedModel.cuda(self.device)
                except:
                    pass
            elif (self.device == "cpu") and (next(self.parameters()).is_cuda):
                try:
                    self.trainedModel.cpu()
                except:
                    pass

            out = self.trainedModel(x, train)
            out_label = tc.argmax(out, dim=-1)

        return out, out_label

    def compute_window_size(self):
        # Assuming stride = 1 and 3 CNN layers
        self.Wn = (((self.sequence_length - self.kernel_size[0] + 1) - self.kernel_size[1] + 1) - self.kernel_size[
            2] + 1)

class MulticlassModelContainer():
    def __init__(self, models, device=None):
        assert type(models) == list
        self.models = models
        self.threshold = models[0].threshold
        self.n_models = len(models)
        if device is not None:
            self.device = device
        else:
            self.device = tc.device("cuda:0" if tc.cuda.is_available else "cpu")
        
        if "cuda" in self.device:
            for model in self.models:
                model.device = device
                try:
                    model.cuda(self.device) 
                except:
                    pass
        else:
          for model in self.models:
                model.device = device
                try:
                    model.cpu()
                except:
                    pass
    def predict(self, X, output_logits=True, 
                output_multiclass=True):
        out = tc.zeros((X.shape[0], self.n_models)).to(self.device)
        for i, model in enumerate(self.models):
            out[:, i] = model.predict(X)
            self.models[i] = model
            
        
        if "cuda" in self.device:
            self.out_logits =  out.cpu().detach().numpy()
        else:
            self.out_logits = out.cpu().detach().numpy()
        
        if output_logits and not output_multiclass:
            return  self.out_logits
        elif output_logits and output_multiclass:
            indices_greater = np.where(self.out_logits > self.threshold)[0]
            temp = (self.out_logits > self.threshold).astype(int)
            temp[indices_greater, 0] = np.argmax(self.out_logits[indices_greater], axis=1)+1
            self.out_labels = temp[:, 0].flatten()
            return self.out_logits, self.out_labels

        else:
            # out = tc.argmax(out, dim=1) + 1
            self.out_labels = (self.out_logits > self.threshold).astype(int)
            return self.out_labels
    
    def get_proba_time(self):
        proba_time = tc.zeros((self.models[0].proba_time.shape[0],
                               self.models[0].proba_time.shape[1],
                               self.n_models))
        for i, model in enumerate(self.models):
            proba_time[:, :, i] = model.proba_time.squeeze(2)
            
        self.proba_time = proba_time