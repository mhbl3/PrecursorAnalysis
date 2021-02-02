###### Loading libraries
#%%

import os
from ..precursorModels.modelProcessing import *
# from utils import *
import json
import time
import torch
import ast
import numpy as np 
import itertools as it
from tqdm import tqdm



###### Class Definition
#%%
class model_trial():
    def __init__(self, trial_directory, data_model, kwargs):
        self.trial_directory = trial_directory
        # Check if folder exist and if the run was completed otherwise creates a new folder or run if run incomplete
        if os.path.exists(self.trial_directory):
            if os.path.exists("completed.txt"): # text file that says the run was completed
                self.run_trial = False
            else:
                self.run_trial = True
        else:
            os.makedirs(self.trial_directory)
            self.run_trial = True

        self.lr = kwargs["learning_rate"]
        self.l2 = kwargs["l2"]
        self.ks = kwargs["kernel_size"]
        self.b_size_percent = kwargs["batch_size_percent"]
        self.out_channels = kwargs["n_filters"]
        self.gru_hidden_size = kwargs["gru_hidden_size"]
        self.model_type = kwargs["model_type"]
        self.optimizer = kwargs["optimizer"]
        self.epochs = kwargs["epochs"]
        self.dropout = kwargs["dropout"]
        self.n_classes = kwargs["n_classes"]
        self.aggregation = kwargs["aggregation"]

        dm = data_model
        self.header = dm.header
        
        if self.n_classes==1:
            self.task = "binary"
        else:
            self.task = "multiclass"

        self.init_dict = kwargs

        with open(os.path.join(self.trial_directory, "Hyperparams.txt"), "w") as f:
            f.write(json.dumps(kwargs))
        self.start_time = time.time()

    def init_my_model(self, input_size, sequence_length, device):
        if self.model_type == "imdope_binary":
            self.trial_model = precursor_model_BN(input_size=input_size,
                                  sequence_length=sequence_length,
                                  kernel_size=self.ks,
                                  dropout=self.dropout,
                                  n_filters=self.out_channels,
                                  hidden_size=self.gru_hidden_size,
                                  fourth_layer="cnn",
                                  device=device)

        elif self.model_type== "imdope_mc":
            self.trial_model= precursor_model_BN_mc(input_size=input_size,
                                  sequence_length=sequence_length,
                                  kernel_size=self.ks,
                                  dropout=self.dropout,
                                  n_classes=self.n_classes,
                                  n_filters=self.out_channels,
                                  hidden_size=self.gru_hidden_size,
                                  fourth_layer="cnn",
                                  device=device)

        elif self.model_type == "lr_binary":
            warnings.warn("Feature importance is not supported")
            MILLR(input_dim=input_size,
                  flight_length=sequence_length,
                  device=device,
                  aggregation=self.aggregation)

    def fit(self, load_trained_model= False, **kwargs):
        """
        pass X_train, y_train, l2, learning_rate, num_epochs
        :param kwargs:
        :return:
        """
        # Saving data
        X_train = kwargs["X_train"]
        self.X_train = X_train
        kwargs["X_train"] = kwargs["X_train"][:, :, :-2]
        y_train = kwargs["y_train"]
        self.y_train = y_train

        batch_size = int(self.b_size_percent*kwargs["X_train"].shape[0])
        start_time_training = time.time()
        if not load_trained_model:
            if not os.path.exists(os.path.join(self.trial_directory, "trial_model.pt")):
              print("Starting to fit model!")
              try:
                 self.trial_model.fit(batch_size=batch_size, clf= self.trial_model, **kwargs)
              except RuntimeError as rte:
                  print(rte)
                  print("Reducing mini-batch size to 2.5% of input size (GPU RAM management)")
                  try:
                      batch_size = int(.10*kwargs["X_train"].shape[0])
                      self.trial_model.fit(batch_size=batch_size, clf=self.trial_model, **kwargs)
                      self.init_dict["batch_size_percent"] = 0.025
                      with open(os.path.join(self.trial_directory, "Hyperparams.txt"), "w") as f:
                         f.write(json.dumps(self.init_dict))
                  except RuntimeError as rte:
                      print(rte)
                      print("Further reducing mini-batch size, try considering mini-batch size less than 2% of training size if ths fails")
                      batch_size = int(.05*kwargs["X_train"].shape[0])
                      self.trial_model.fit(batch_size=batch_size, clf=self.trial_model, **kwargs)
                      self.init_dict["batch_size_percent"] = 0.02
                      with open(os.path.join(self.trial_directory, "Hyperparams.txt"), "w") as f:
                        f.write(json.dumps(self.init_dict))
            else:
              print("Loading model!")
              with open(os.path.join(self.trial_directory, "trial_model.pt"), "rb") as f:
                self.trial_model = torch.load(f)
        else:
            print("Loading model!")
            with open(os.path.join(self.trial_directory, "trial_model.pt"), "rb") as f:
                self.trial_model = torch.load(f)
        self.training_time = time.time() - start_time_training
        
        self.trial_model.save(os.path.join(self.trial_directory, "trial_model"))

        # Basic Metrics
        # self.CM_train = self.trial_model.CM
        # self.balanced_accuracy_train = self.trial_model.b_acc[-1]
        # self.f1_train = self.trial_model.f1[-1]
        loss_train = self.trial_model.hist[-1]
#         try:
#           self.trial_model.cuda()
#           self.trial_model.device = "cuda:0"
#         except:
#           pass 
#         prediction_logits = self.trial_model.predict(kwargs["X_train"])
#         try:
#           prediction_logits = prediction_logits.cpu().detach().numpy()
#         except:
#           prediction_logits = prediction_logits.detach().numpy()
        prediction_logits = self.trial_model.evaluate_model(mode="train", )[0].cpu().detach().numpy()
        if self.n_classes == 1:
            prediction_train = (prediction_logits > self.trial_model.threshold).astype(int)
        else:
            prediction_train = prediction_logits
        self.summary_results_train = pd.DataFrame(columns=["CM", "Balanced_Acc", "f1"],
                                            index=range(len(X_train)))
#         confusion_matrix(y_train, prediction_train).flatten()
        self.summary_results_train.at[:, "CM"] = [confusion_matrix(y_train, prediction_train).flatten()] * len(X_train)
        self.summary_results_train.loc[:, "Balanced_Acc"] = balanced_accuracy_score(y_train, prediction_train)
        if self.n_classes == 1:
            self.summary_results_train.loc[:, "f1"] = f1_score(y_train, prediction_train, average="binary")
        else:
            self.summary_results_train.loc[:, "f1"] = f1_score(y_train, prediction_train, average="weighted")
        self.summary_results_train.loc[:, "class"] = y_train
        self.summary_results_train.loc[:, "pred_class"] = prediction_train
        self.summary_results_train.loc[:, "loss"] = loss_train



    def evaluate_basic_metrics(self, X_val, y_val):
        # Saving data
        X_val = X_val.astype(np.float64)
        self.trial_model.x_test = X_val[:, :, :-2]
        self.X_val = X_val
        self.y_val = y_val 

        # Eval
        prediction_logits, _ = self.trial_model.evaluate_model(mode="val",)
        prediction_logits = prediction_logits.detach().numpy()
        if self.n_classes == 1:
            logic = prediction_logits > self.trial_model.threshold
            prediction_val = np.asarray(logic).astype(int)
        else:
            prediction_val = prediction_logits
        
        # Basic Metrics
#         self.prediction_val = (self.prediction_val > self.trial_model.threshold).astype(int)
#         CM_val = confusion_matrix(y_val, prediction_val)
#         balanced_accuracy_val = balanced_accuracy_score(y_val, prediction_val)
#         f1_val = f1_score(y_val, prediction_val, average="binary")
        
        self.summary_results_val = pd.DataFrame(columns=["CM", "Balanced_Acc", "f1"],
                                                  index=range(len(X_val)))
        self.summary_results_val.at[:, "CM"] = [confusion_matrix(y_val, prediction_val).flatten()] * len(X_val)
        self.summary_results_val.loc[:, "Balanced_Acc"] = balanced_accuracy_score(y_val, prediction_val)
        if self.n_classes == 1:
            self.summary_results_val.loc[:, "f1"] = f1_score(y_val, prediction_val, average="binary")
        else:
            self.summary_results_val.loc[:, "f1"] = f1_score(y_val, prediction_val, average="weighted")
        self.summary_results_val.loc[:, "class"] = y_val
        self.summary_results_val.loc[:, "pred_class"] = prediction_val
        
    def evaluate_ADOPT_overlap(self, path_to_ADOPT_run=None):

        # Comparing overlap with ADOPT
        if path_to_ADOPT_run is None:
            path_to_ADOPT_run = ["/content/drive/My Drive/MS Thesis/ADOPT/HS_case/output/run_1",
                                 "/content/drive/My Drive/MS Thesis/ADOPT/HP_case/output/run_1"]

        # Go through training data
#         self.trial_model.predict(self.X_train[:, :, :-2]) gets proba
#         proba_time = self.trial_model.proba_time.detach().numpy() 
        y_hat, proba_time = self.trial_model.evaluate_model(mode="train",
            get_important_features=False)
        # self.deviation_df_train = pd.DataFrame(columns=["N_time_steps_deviation", "miles_deviation"],
        #                                  index=range(self.X_train.shape[0]))
        self.summary_results_train["N_time_steps_deviation"] = np.nan
        self.summary_results_train["miles_deviation"] = np.nan
        idx_adverse = np.where(self.y_train!=0)[0]
        deviation_lst = []
        deviation_miles_lst = [] 
        flight_id_lst = [] 
        path = ""
        
        for e, flight_idx in enumerate(tqdm(idx_adverse)):
            temp_proba_time = window_padding(proba_time[flight_idx, :].flatten()
                            , self.X_train.shape[1])
            # get ADOPT score
            flight_id = int( self.X_train[flight_idx, 0, -1])
            if self.n_classes ==1:
                adopt_idx = 0
            else:
                adopt_idx = max(0, int(y_hat[e]-1)) # assumes normal is the same for both
            path = os.path.join(path_to_ADOPT_run[adopt_idx], "parameter_graphs", "precursor_score__data_{}.pkl".format(flight_id))
            if os.path.exists(path):
              with open(path, "rb") as f :
                temp_ADOPT_proba_Time = pkl.load(f)

            # df = pd.DataFrame({"ADOPT": temp_ADOPT_proba_Time, "MHCNN": temp_proba_time},
            #                   index=np.arange(20, -0.25, -0.25))

            # if (df.MHCNN > self.trial_model.threshold).any():
            #     deviation = len(df[df.MHCNN > self.trial_model.threshold]) - len(
            #         df[(df.ADOPT > self.trial_model.threshold) & (df.MHCNN > self.trial_model.threshold)])
            # else:
            #     deviation = np.nan
              if np.where(temp_proba_time > self.trial_model.threshold)[0].shape > 0:
                deviation = np.where(temp_proba_time > self.trial_model.threshold)[0].shape[0] - \
                    np.where((temp_ADOPT_proba_Time > self.trial_model.threshold) & ( temp_proba_time > self.trial_model.threshold))[0].shape[0]
              else:
                deviation = np.nan 
            else:
              deviation = np.nan

            deviation_lst.append(deviation)
            deviation_miles_lst.append(deviation * 0.25)
            flight_id_lst.append(flight_id)
#         print(len(idx_adverse))
#         print(len(np.asarray(deviation_lst)))
#         print(len(deviation_lst))
        self.summary_results_train.loc[idx_adverse, "N_time_steps_deviation"] = np.asarray(deviation_lst)
        self.summary_results_train.loc[idx_adverse, "miles_deviation"] = np.asarray(deviation_miles_lst)
        self.summary_results_train.loc[idx_adverse, "flight_id"] = np.asarray(flight_id_lst)

        
        # Go through validation data
#         self.trial_model.predict(self.X_val[:, :, :-2])  gets proba
#         proba_time = self.trial_model.proba_time.detach().numpy()

        y_hat, proba_time = self.trial_model.evaluate_model(mode="val",
            get_important_features=False)
        # self.deviation_df_train = pd.DataFrame(columns=["N_time_steps_deviation", "miles_deviation"],
        #                                  index=range(self.X_train.shape[0]))
        self.summary_results_val["N_time_steps_deviation"] = np.nan
        self.summary_results_val["miles_deviation"] = np.nan
        idx_adverse = np.where(self.y_val != 0)[0]
        deviation_lst = []
        deviation_miles_lst = [] 
        flight_id_lst = [] 
        
        for e, flight_idx in enumerate(tqdm(idx_adverse)):
            temp_proba_time = window_padding(proba_time[flight_idx, :].flatten()
                                             , self.X_val.shape[1])
            # get ADOPT score
            flight_id = int(self.X_val[flight_idx, 0, -1])
            if self.n_classes ==1:
                adopt_idx = 0
            else:
                adopt_idx = max(0, int(y_hat[e]-1)) # assumes normal is the same for both
            path = os.path.join(path_to_ADOPT_run[adopt_idx], "parameter_graphs", "precursor_score__data_{}.pkl".format(flight_id))
            if os.path.exists(path):
              with open(path, "rb") as f:
                  temp_ADOPT_proba_Time = pkl.load(f)

            # df = pd.DataFrame({"ADOPT": temp_ADOPT_proba_Time, "MHCNN": temp_proba_time},
            #                   index=np.arange(20, -0.25, -0.25))

            # if (df.MHCNN > self.trial_model.threshold).any():
            #     deviation = len(df[df.MHCNN > self.trial_model.threshold]) - len(
            #         df[(df.ADOPT > self.trial_model.threshold) & (df.MHCNN > self.trial_model.threshold)])
            # else:
            #     deviation = np.nan
              if np.where(temp_proba_time > self.trial_model.threshold)[0].shape > 0:
                  deviation = np.where(temp_proba_time > self.trial_model.threshold)[0].shape[0] - \
                    np.where((temp_ADOPT_proba_Time > self.trial_model.threshold) & ( temp_proba_time > self.trial_model.threshold))[0].shape[0]
              else:
                  deviation = np.nan
            else:
              deviation = np.nan
                
            deviation_lst.append(deviation)
            deviation_miles_lst.append(deviation * 0.25)
            flight_id_lst.append(flight_id)

        self.summary_results_val.loc[idx_adverse, "N_time_steps_deviation"] =  np.asarray(deviation_lst)
        self.summary_results_val.loc[idx_adverse, "miles_deviation"] =  np.asarray(deviation_miles_lst)
        self.summary_results_val.loc[idx_adverse, "flight_id"] =  np.asarray(flight_id_lst)

    def evaluate_best_feature_overlap(self, path_to_ADOPT_run=None):
        if path_to_ADOPT_run is None:
            path_to_ADOPT_run = ["/content/drive/My Drive/MS Thesis/ADOPT/HS_case/output/run_1",
                                 "/content/drive/My Drive/MS Thesis/ADOPT/HP_case/output/run_1"]

        # Training
#         self.trial_model.predict(self.X_train[:, :, :-2])
        self.trial_model.header = self.header 
        y_hat, _, top_features_list, top_features_values_list = self.trial_model.evaluate_model(mode="train",
            get_important_features=True)
        self.summary_results_train["n_common_features"] = np.nan
        self.summary_results_train["top_features"] = np.nan
        self.summary_results_train["top_features_ADOPT"] = np.nan
        idx_adverse = np.where(self.y_train!=0)[0]
        self.ADOPT_not_found = []
        print("Looking for ADOPT Data")
        for e, flight_idx in enumerate(tqdm(idx_adverse)):
            flight_id = int(self.X_train[flight_idx, 0, -1])
            if self.n_classes ==1:
                adopt_idx = 0
            else:
                adopt_idx = max(0, int(y_hat[e]-1)) # assumes normal is the same for both
#             self.trial_model.show_feature_importance(self.header, flight_idx, plot=False)
            top_features = top_features_list[flight_idx]
            top_features_values = top_features_values_list[flight_idx]
            # Get Adopt top 10
            path_ADOPT_train = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking", f"Anomalous_Train_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_train_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking", f"Anomalous_Train_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")
            path_ADOPT_test = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking", f"Anomalous_Test_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_test_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking", f"Anomalous_Test_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")
            path_ADOPT_val = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                           f"Anomalous_Validation_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_val_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                             f"Anomalous_Validation_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")
            path = ""
            if os.path.exists(path_ADOPT_train):
                path = path_ADOPT_train
            elif os.path.exists(path_ADOPT_train_2):
                path = path_ADOPT_train_2
            elif os.path.exists(path_ADOPT_test):
                path = path_ADOPT_test
            elif os.path.exists(path_ADOPT_test_2):
                path = path_ADOPT_test_2
            elif os.path.exists(path_ADOPT_val):
                path = path_ADOPT_val
            elif os.path.exists(path_ADOPT_val_2):
                path = path_ADOPT_val_2

            if path == "":
                self.ADOPT_not_found.append(flight_id)
                temp_ADOPT_ranking = np.nan
                points = 0
                temp_ADOPT_ranking_values = np.nan
            else:
                with open(path, "rb") as f:
                    temp_ADOPT_ranking = pkl.load(f)
#                 print(path)
#                 print(temp_ADOPT_ranking)
#                 print(top_features)
#                 print(temp_ADOPT_ranking)
                if isinstance(top_features, float):
                  top_features = [top_features]
                points = sum([1 for i in top_features if i in temp_ADOPT_ranking])
                with open(path.split(".pkl")[0]+"_score.pkl", "rb") as f:
                    temp_ADOPT_ranking_values = pkl.load(f)


            self.summary_results_train.loc[flight_idx, "n_common_features"] = points
            self.summary_results_train.loc[flight_idx, "top_features"] = str(top_features)
            self.summary_results_train.loc[flight_idx, "top_features_values"] = str(top_features_values)
            self.summary_results_train.loc[flight_idx, "top_features_ADOPT"] = str(temp_ADOPT_ranking)
            self.summary_results_train.loc[flight_idx, "top_features_ADOPT_values"] = str(temp_ADOPT_ranking_values)

        # Validation
#         self.trial_model.predict(self.X_val[:, :, :-2])
        y_hat, _, top_features_list, top_features_values_list = self.trial_model.evaluate_model(mode="val",
            get_important_features=True)
        self.summary_results_val["n_common_features"] = np.nan
        self.summary_results_val["top_features"] = np.nan
        self.summary_results_val["top_features_ADOPT"] = np.nan
        idx_adverse = np.where(self.y_val==1)[0]

        for e, flight_idx in enumerate(tqdm(idx_adverse)):
            flight_id = int(self.X_val[flight_idx, 0, -1])
            if self.n_classes ==1:
                adopt_idx = 0
            else:
                adopt_idx = max(0, int(y_hat[e]-1)) # assumes normal is the same for both
#             self.trial_model.show_feature_importance(self.header, flight_idx, plot=False)
#             top_features = self.trial_model.sorted_features
#             top_features_values = self.trial_model.sorted_features_values
            top_features = top_features_list[flight_idx]
            top_features_values = top_features_values_list[flight_idx]
            # Get Adopt top 10
            path = ""
            path_ADOPT_train = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                            f"Anomalous_Train_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_train_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                              f"Anomalous_Train_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")
            path_ADOPT_test = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                           f"Anomalous_Test_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_test_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                             f"Anomalous_Test_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")
            path_ADOPT_val = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                          f"Anomalous_Validation_ranking_data_{flight_id}.pdf_data_{flight_id}.pkl")
            path_ADOPT_val_2 = os.path.join(path_to_ADOPT_run[adopt_idx], "feature_ranking",
                                            f"Anomalous_Validation_ranking_data_{flight_id}_precursor_event_0.pdf_data_{flight_id}_precursor_event_0.pkl")

            if os.path.exists(path_ADOPT_train):
                path = path_ADOPT_train
            elif os.path.exists(path_ADOPT_train_2):
                path = path_ADOPT_train_2
            elif os.path.exists(path_ADOPT_test):
                path = path_ADOPT_test
            elif os.path.exists(path_ADOPT_test_2):
                path = path_ADOPT_test_2
            elif os.path.exists(path_ADOPT_val):
                path = path_ADOPT_val
            elif os.path.exists(path_ADOPT_val_2):
                path = path_ADOPT_val_2

            if path == "":
                self.ADOPT_not_found.append(flight_id)
                temp_ADOPT_ranking = np.nan
                points = 0
                temp_ADOPT_ranking_values = np.nan
            else:
                with open(path, "rb") as f:
                    temp_ADOPT_ranking = pkl.load(f)
                if isinstance(top_features, float):
                  top_features = [top_features]
                points = sum([1 for i in top_features if i in temp_ADOPT_ranking])
                with open(path.split(".pkl")[0]+"_score.pkl", "rb") as f:
                    temp_ADOPT_ranking_values = pkl.load(f)

            self.summary_results_val.loc[flight_idx, "n_common_features"] = points
            self.summary_results_val.loc[flight_idx, "top_features"] = str(top_features)
            self.summary_results_val.loc[flight_idx, "top_features_values"] = str(top_features_values)
            self.summary_results_val.loc[flight_idx, "top_features_ADOPT"] = str(temp_ADOPT_ranking)
            self.summary_results_val.loc[flight_idx, "top_features_ADOPT_values"] = str(temp_ADOPT_ranking_values)

    def save_all(self):
        train_csv = os.path.join(self.trial_directory, "train_results.csv")
        val_csv = os.path.join(self.trial_directory, "val_results.csv")
        self.summary_results_train.to_csv(train_csv)
        self.summary_results_val.to_csv(val_csv)

        path_text = os.path.join(self.trial_directory, "completed.txt")

        # self.trial_model.save(os.path.join(self.trial_directory, "trial_model"))
        with open(os.path.join(self.trial_directory, "trial_model_class.pkl"), "wb") as f:
            pkl.dump(self, f)

        with open(path_text, "a") as f:
            f.write(f"Number of epochs: {self.trial_model.n_epochs}\n")
            f.write(f"Training size: {self.X_train.shape}\n")
            f.write(f"Number of parameters: {self.trial_model.count_parameters()}\n")
            f.write(f"Training time (min): {self.training_time/60}\n")
            f.write("Total time taken for combination (min): {}".format((time.time()-self.start_time)/60))


def tuning_wraper(data, input, load_trained_model=False,
                  run_methods=[True, True, True], path_to_ADOPT_run=None):
    X_train, y_train = data["X_train"], data["y_train"]
    dm = data["data_model"]
    X_test, y_test = data["X_test"], data["y_test"]

    list_params = input["list_params"]
    combination = input["combination"]
    path = input["path"]
    device = input["device"]

    temp_dict = dict(zip(list_params, combination))

    mt = model_trial(path, dm, temp_dict)
    print("Model initialization!")
    mt.init_my_model(X_train.shape[2] - 2, X_train.shape[1], device)
    mt.fit(X_train=X_train, y_train=y_train, l2=temp_dict["l2"],
           learning_rate=temp_dict["learning_rate"],
            class_weight = temp_dict.pop("class_weight", None),
            use_stratified_batch_size = temp_dict.pop("use_stratified_batch_size", False),
           num_epochs=temp_dict["epochs"], load_trained_model=load_trained_model)
    print("Evaluating model!")
    mt.evaluate_basic_metrics(X_test, y_test)
    
    if run_methods[0]:
        print("Evaluating Time Overlaps")
        if mt.n_classes >1:
            warnings.warn("Inaccurate feature importance of imdope may be calculated. Change n_classes to 1")
        mt.evaluate_ADOPT_overlap(path_to_ADOPT_run)
        
    if run_methods[1]:
        print("Evaluating Feature Overlaps")
        if mt.n_classes >1:
            warnings.warn("Inaccurate feature importance of imdope may be calculated. Change n_classes to 1")
        mt.evaluate_best_feature_overlap(path_to_ADOPT_run)
        
    if run_methods[2]:
        print("Saving everyting!")
        mt.save_all()
        
    torch.cuda.empty_cache()
    
    
def format_inputs_tuning_wraper(X_train, y_train, X_test, y_test, params, data_model, path_thesis,
 ADOPT_path=None, run_methods=[True, True, True], load_trained_model=False, device="cuda:0",
                                search_directory="search", run_tuning_wraper=True):
    list_params = []
    procs = []
    combs = [] 
    data = {"X_train": X_train, "y_train": y_train,
        "data_model": data_model,
        "X_test": X_test, "y_test": y_test}
    if type(params) == str:
        params = json.load(params)
    for name, _ in params.items():
        list_params.append(name)

    all_combinations = list(it.product(*(params[name] for name in list_params)))
    all_combinations = [list(tuple) for tuple in all_combinations]
    [i.insert(0, counter) for counter, i in enumerate(all_combinations)] # insert index of file

    # %% trial prep
    if not os.path.exists(os.path.join(path_thesis,"Data", search_directory)):
        os.makedirs(os.path.join(path_thesis,"Data", search_directory))

    with open(os.path.join(path_thesis,"Data", f"{search_directory}/all_combinations.txt"), 'a') as filehandle:
        filehandle.write(('%s\n' % str(list_params)))
        for listitem in all_combinations:
            filehandle.write('%s\n' % str(listitem))

    start = time.time()

    for run, combination in enumerate(all_combinations):
        path = os.path.join(path_thesis,"Data", search_directory, f"Combination_{combination[0]}")
        if not load_trained_model:
            if os.path.exists(path):
                if os.path.exists(os.path.join(path,"completed.txt")):
                    continue
            else:
                os.makedirs(path)
        else:
            if not os.path.exists(os.path.join(path,"trial_model.pkl")):
                continue

        myinput = {"combination": combination[1:],
                    "dm": data_model, "list_params": list_params,
                    "path": path, "device": device}
        combs.append(myinput)

    if run_tuning_wraper:
        for comb in combs:
            tuning_wraper(data, comb, load_trained_model, path_to_ADOPT_run= ADOPT_path, run_methods=run_methods)
    else:
        return data, combs
    
# Evaluation of hyper parameter search
def train_test_results_selection(df, combination, path_data,
                                 search_folder="search", train_results=True):
    if train_results:
        combs_dir = f"{combination}/train_results.csv"
        full_path = os.path.join(path_data, search_folder, combs_dir)
        temp_df = pd.read_csv(full_path)
        df.b_acc_train.loc[combination] = temp_df.Balanced_Acc.iloc[0]
        df.f1_train.loc[combination] = temp_df.f1.iloc[0]
        deviations = temp_df.N_time_steps_deviation
        df.deviation_mean_train.loc[combination] = deviations.mean()
        df.deviation_std_train.loc[combination] = deviations.std()
    else:
        combs_dir = f"{combination}/val_results.csv"
        full_path = os.path.join(path_data, search_folder, combs_dir)
        temp_df = pd.read_csv(full_path)
        df.b_acc_val.loc[combination] = temp_df.Balanced_Acc.iloc[0]
        df.f1_val.loc[combination] = temp_df.f1.iloc[0]
        deviations = temp_df.N_time_steps_deviation
        df.deviation_mean_val.loc[combination] = deviations.mean()
        df.deviation_std_val.loc[combination] = deviations.std()

    tops = temp_df.top_features
    tops_Values = temp_df.top_features_values
    top_ADOPT = temp_df.top_features_ADOPT
    top_ADOPT_values = temp_df.top_features_ADOPT_values

    mse_list = []
    feats = tops[pd.notna(tops)].iloc[0]
    se_np = np.zeros((len(make_list(feats)), len(tops)))
    flag = True
    list_extra = [] 
    for i, tupl in enumerate(zip(tops_Values, tops)):
        top_val, top = tupl
        # Check if values or precursors are ok
        mse_calc_ok = False
        if isinstance(top, str) and isinstance(top_val, str):
            if ("None" not in top ) and ("None" not in top_val):
                mse_calc_ok = True
        if temp_df.pred_class[i] != 0:
            if (pd.notna(top_ADOPT.iloc[i])) and (pd.notna(top_ADOPT_values.iloc[i])) and (mse_calc_ok):
                mse, se, _, _, list_extra= feature_ranking_mse(top_val,
                                                   top,
                                                   top_ADOPT_values.iloc[i],
                                                   top_ADOPT.iloc[i])
                mse_list.append(mse)
                if flag:
                    se_np = np.zeros((len(se), len(tops)))
                    flag = False
                se_np[:, i] = se
        else:
          mse_list.append(np.nan)
          se_np[:, i] = np.nan
    mse_combination = np.nanmean(np.asarray(mse_list))
    if train_results:
        df.loc[combination, "MSE"] = mse_combination
    else:
        df.loc[combination, "MSE"] = mse_combination

    se_combination = np.nanmean(se_np, axis=1)
    sorted_features = [feat for feat in sorted(make_list(feats), key=lambda x: x.lower())]
    for feature in list_extra:
      if feature in sorted_features:
        sorted_features.remove(feature)
    index_ = [index for index, col in enumerate(sorted_features) if col not in df.columns]

    list_features = np.asarray(sorted_features)
    if index_ != []:
        for col in list_features[index_]:
            df[col] = 0
    if train_results:
      if "None" not in list_features[0]: #by product of appended None in the evaluate_model() in modelProcressing
        df.loc[combination, list_features] = se_combination
    else:
      if "None" not in list_features[0]:
        df.loc[combination, list_features] = (df.loc[combination, list_features] + se_combination) / 2

    return df

def make_list(string_features):
    return [i for i in string_features.split("'") if (i!=" ") and (i!= '[') and (i!=']') and (i.replace(" ","") !="\n" and (i.replace(" ","")!=","))]

def feature_ranking_mse(myRanking, myRankingColumns, adoptRanking, adoptColumns):
    if type(myRanking) == str:
        try:
            myRanking = ast.literal_eval(myRanking.replace(" ", ","))
        except SyntaxError:
            temp = myRanking.replace(" ", ",").split(",")
            temp = [x for x in temp if "." in x]
            temp = [x.replace("[", "") for x in temp ]
            temp = [x.replace("]", "") for x in temp ]
            temp = [x.replace("\n", "") for x in temp]
            myRanking = [ast.literal_eval(x) for x in temp]

    if type(myRankingColumns) == str:
        myRankingColumns = make_list(myRankingColumns)
    if type(adoptRanking) == str:
        adoptRanking = ast.literal_eval(adoptRanking)
    if type(adoptColumns) == str:
        adoptColumns = make_list(adoptColumns)

    # Sort features in alphabatical order, used that order to sort their scores
    if isinstance(myRanking, float):# In case a nan is received instead of a list
      myRanking = [myRanking] * len(myRankingColumns)

    if (not all([True if i in myRankingColumns else False for i in adoptColumns])) or (not all([True if i in adoptColumns  else False for i in myRankingColumns])):
    # if len(myRankingColumns) != len(adoptColumns):
        # Reduce ADOPT list
        idx = [i for i, col in enumerate(adoptColumns) if col in myRankingColumns]
        adoptColumns = list(np.asarray(adoptColumns)[idx])
        adoptRanking = list(np.asarray(adoptRanking)[idx])
        # In case there is still a different length
        list_extra = [i for i in myRankingColumns if i not in adoptColumns]
        idx_list_extra = [j for j,i in enumerate(myRankingColumns) if i in adoptColumns]
        myRanking = list(np.asarray(myRanking)[idx_list_extra])
        myRankingColumns = list(np.asarray(myRankingColumns)[idx_list_extra])

    myRankingCol = [col for col, _ in sorted(zip(myRankingColumns, myRanking), key=lambda x: x[0].lower())]
    myRankingVal = np.asarray([val for _, val in sorted(zip(myRankingColumns, myRanking), key=lambda x: x[0].lower())])
    myRanking_tuple = [x for x in sorted(zip(myRankingColumns, myRanking), key=lambda x: x[0].lower())]
    # Normalize score btw 0-1
    try:
      myRankingVal = (myRankingVal - myRankingVal.min()) / (myRankingVal.max() - myRankingVal.min())
    except:
      return np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        adoptColumns.remove("Unnamed: 0")
    except:
        pass




    # Sort features in alphabetical order, used that order to sort their scores
    #     adoptCol = [col for col,_ in sorted(zip(adoptColumns, adoptRanking), key= lambda x: x[0].lower())] # same as line 12
    ADOPTRankingVal = np.asarray(
        [val for _, val in sorted(zip(adoptColumns, adoptRanking), key=lambda x: x[0].lower())])
    ADOPT_tuple = [x for x in sorted(zip(adoptColumns, adoptRanking), key=lambda x: x[0].lower())]
    # Normalize score btw 0-1
    if len(ADOPTRankingVal) == 0: 
      ADOPTRankingVal = np.asarray([np.nan]*len(myRankingVal))
    ADOPTRankingVal = (ADOPTRankingVal - ADOPTRankingVal.min()) / (ADOPTRankingVal.max() - ADOPTRankingVal.min())

    # Compute mse for flight
    mse = ((myRankingVal - ADOPTRankingVal) ** 2).mean()
    # Compute squared error for feature
    se = (myRankingVal - ADOPTRankingVal) ** 2

    return mse, se, myRanking_tuple, ADOPT_tuple, list_extra

def rank_combinations(counter_df, metrics_of_interest, actions, list_hyperparams, weights=None):
    if weights==None:
        n_metrics = len(metrics_of_interest)
        weights = np.zeros((n_metrics,1))
        weights[:,0] = 1/n_metrics

    m_counter_df = counter_df.dropna()
    comb_ranking = pd.DataFrame(data={"Points": np.zeros(len(m_counter_df))},
                                index=m_counter_df.index)

    for i, it in enumerate(zip(metrics_of_interest, actions)):
        metric, action = it
        if action == "min":
            ascendence = False
        elif action == "max":
            ascendence = True
        else:
            raise ValueError("actions must be max or min")

        sorted_combination = m_counter_df.sort_values(by=metric, ascending=ascendence).index
        comb_ranking["Points"] = comb_ranking["Points"] + \
                                 weights[i] * np.asarray(
            [np.where(comb == sorted_combination.values)[0][0] + 1 for comb in comb_ranking.index.values])

    return comb_ranking.sort_values(by="Points", ascending=False).merge(m_counter_df[list_hyperparams],
                                                                left_index=True,
                                                                right_index=True)

def create_combination_df(path_data, n_combinations, search_dir = "search"):
    # Get all combinations
    path_all_comb_txt = os.path.join(path_data, search_dir, "all_combinations.txt")
    combs = []
    with open(path_all_comb_txt, "r") as f:
        for i, x in enumerate(f):
            try:
                combs.append(ast.literal_eval(x))
            except:
                pass
    combs = combs[:n_combinations+1]
    # Retrieve each metric
    df_index = ["Combination_{}".format(i) for i in range(n_combinations)]
    counter_df = pd.DataFrame(columns=["b_acc_train", "f1_train", "deviation_mean_train", "deviation_std_train",
                                       "b_acc_val", "f1_val", "deviation_mean_val", "deviation_std_val"],
                              index=df_index)

    for idx_param, param in enumerate(combs[0]):
        counter_df[param] = np.nan
    for idx_param, param in enumerate(combs[0]):
        for i in range(len(combs) - 1):
            counter_df[param].iloc[i] = str(combs[i + 1][idx_param + 1])
    for i, comb in enumerate(counter_df.index):
        counter_df = train_test_results_selection(counter_df, comb, path_data, search_dir)
        counter_df = train_test_results_selection(counter_df, comb, path_data, search_dir, False)

    return counter_df, combs

def TOPSIS(data, metrics, benefit_cost, list_hyperparams, weights=None):
    if weights==None:
        n_metrics = len(metrics)
        weights = np.zeros((n_metrics,1))
        weights[:,0] = 1/n_metrics
    data_original = data.dropna().copy()
    data = data[metrics].dropna().copy()
    ideal_pos= np.empty(data.shape[1])
    ideal_neg= np.empty(data.shape[1])
    data["score"] = 0
    for i,col in enumerate(data.columns[0:-1]):
        data.loc[:,col] =(data.loc[:,col] /np.sqrt(np.sum(data.loc[:, col]**2))) * weights[i]
        if benefit_cost[i]=="max":
            ideal_pos[i] = np.max(data.loc[:,col])
            ideal_neg[i] = np.min(data.loc[:,col])
        elif benefit_cost[i]=="min":
            ideal_pos[i] = np.min(data.loc[:,col])
            ideal_neg[i] = np.max(data.loc[:,col])
    for j in range(data.shape[0]):
        S_pos= np.sqrt(np.sum((data.iloc[j, :-1].values- ideal_pos)**2))
        S_neg= np.sqrt(np.sum((data.iloc[j, :-1]- ideal_neg)**2))
        data.loc[data.index[j], "score"] = S_neg/(S_pos+S_neg)
    index = np.argmax(data.loc[:, "score"])
    print("Best Alternative is: {}".format(data.index[index]))
    out_data = data_original[list_hyperparams]
    out_data["score"] = data["score"]
    return out_data.sort_values(by="score", ascending=False)

# if __name__ == '__main__':
#     counter_df, combs = create_combination_df("../", 1, "SciTech_Search")
#     print(combs)