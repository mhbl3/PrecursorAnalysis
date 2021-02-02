import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek

import pickle as pkl
import time
import json
import os
from scipy.stats import norm
import torch


class DataContainer():
    """
    Class for containing data for the precursor analysis.
    it contains methods that I found useful for the data processing part
    """
    def __init__(self, data, **kwargs):
        """
        Initialze the container .

        Parameters
        ----------
        data : pd.DataFrame, list
            if pd.DataFrame the data frame will assigned to dataContainer.df, if it's a list the first index must correspond to the nominal directory and the second to the adverse directory.
            The csv in the folders will be processed.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(data) == str and "csv" in data.lower():
            self.df = pd.read_csv(data, **kwargs)
        elif type(data) == pd.DataFrame:
            self.df = data
        elif type(data)== np.ndarray:
            self.df = pd.DataFrame(data, **kwargs)
        elif type(data)==str and "pkl" in data.lower():
            with open(data, "rb") as file:
                data = pkl.load(file)
                if type(data)==pd.DataFrame:
                    self.df = data
                elif type(data) == np.ndarray:
                    self.df = pd.DataFrame(data, **kwargs)
                else:
                    raise Exception("Insert dataframe df in the data container by using DataContainer().df = df")
        elif type(data) == str and "json" in data.lower():
            with open(data, "rb") as file:
                data = json.loads(file)
            self.df = pd.DataFrame(data, **kwargs)
        elif type(data) == list:
            nominal_directory = data[0]
            anomalous_directory = data[1]
            fix_path = kwargs.pop("fix_path", False)
            if fix_path:
                new_path = os.getcwd().replace("/", "\\").split("\\")
                idx = [i for i, e in enumerate(new_path) if e =="PrecursorAnalysis"][0]
                tmp = new_path[0]
                for i, e in enumerate(new_path[1:], 1):
                    if i <=idx:
                        tmp = tmp +"/"+e
                new_path = tmp
                self.fixed_data_path = new_path
                joined_nom  = os.path.join(new_path, nominal_directory.replace("/", "\\"))
                files_nom = os.listdir(joined_nom)
                joined_adv = os.path.join(new_path, anomalous_directory.replace("/","\\"))
                files_anom = os.listdir(joined_adv)
            else:
                files_nom = os.listdir(nominal_directory)
                files_anom = os.listdir(anomalous_directory)
            
            all_flight_list = []
            self.file_not_used = []
            verbose = kwargs.pop("verbose", 0)
            replace_name_target = kwargs.pop("replace_name_target", None)
            random_int = np.random.randint(1, len(files_nom))

            for i, nom in enumerate(files_nom):
                if fix_path:
                    dir_temp = os.path.join(joined_nom, nom)
                else:
                    dir_temp = os.path.join(nominal_directory,nom)
                temp = pd.read_csv(dir_temp)
                if replace_name_target is not None:
                    string_to_use = "Anomaly"
                    temp.rename(columns={f"{replace_name_target}":f"{string_to_use}"}, inplace=True)
                tmp_cols = [i for i in temp.columns.values if i != "Anomaly"]
                if "filename" not in tmp_cols:
                    temp["filename"] = nom
                    tmp_cols.append("filename")
                if "flight_id" not in tmp_cols:
                    temp["flight_id"] = 0 # will be replaced later
                    tmp_cols.append("flight_id")
                tmp_cols.append("Anomaly")
                # Making the flight_id, and anomaly columns is at the end
                temp = temp[tmp_cols]

                if i == 0:
                    if fix_path:
                        tmp_df = pd.read_csv(os.path.join(joined_nom, files_nom[random_int]))
                    else:
                        tmp_df = pd.read_csv(os.path.join(nominal_directory, files_nom[random_int]))
                    self.max_len = len(tmp_df)

                    header = kwargs.pop("header", temp.columns)
                    print("Flight Length set to {}".format(self.max_len))
                    if verbose == 1:
                        print("Keeping only the following columns: {}".format(header))
                temp = temp[header]
                flight_id = nom.split("_")[1].split(".")[0]
                temp.flight_id = int(flight_id) # replace flight id
                if verbose > 1:
                    print(f"Processed Nominal Flight {flight_id}\n")
                if len(temp) == self.max_len:
                    all_flight_list.append(temp)
                else:
                    self.file_not_used.append(dir_temp)
            
            for anom in files_anom:
                if fix_path:
                    dir_temp = os.path.join(joined_adv, anom)
                else:
                    dir_temp = os.path.join(anomalous_directory, anom)
                temp = pd.read_csv(dir_temp)
                if replace_name_target is not None:
                    temp.rename(columns={f"{replace_name_target}":f"{string_to_use}"}, inplace=True)
                tmp_cols = [i for i in temp.columns.values if i != "Anomaly"]
                # Ideally the adverse event data should have the same throughout the whole flight
                if 0 in temp.Anomaly.unique():
                    temp["Anomaly"] = 1
                if "filename" not in tmp_cols:
                    temp["filename"] = anom
                    tmp_cols.append("filename")
                if "flight_id" not in tmp_cols:
                    temp["flight_id"] = 0 # will be replaced later
                    tmp_cols.append("flight_id")
                tmp_cols.append("Anomaly")
                # Making the flight_id, and anomaly columns is at the end
                temp = temp[header]
                flight_id = anom.split("_")[1].split(".")[0]
                temp.flight_id = int(flight_id) # replace flight id
                
                if verbose > 1:
                    print(f"Processed Anomalous Flight {flight_id}\n")                
                if len(temp) == self.max_len:
                    all_flight_list.append(temp)
                else:
                    self.file_not_used.append(dir_temp)
                
                # Concat all flights 
            print("Now Concatenating all flights into a dataframe")
            self.df = pd.concat(all_flight_list)
        else:
            raise Exception("Loading for this data type is not available")
        if not hasattr(self, "header"):
            self.header = self.df.columns

    def create_nominal_adverse_directories(self, path_to_data_directory, anomaly=1, 
                                           create_nominal_files=True, create_adverse_files=True,
                                           **kw):
        """
        Create the directories and fill them with nominal and specified adverse flights data. The files are all csv format.

        Parameters
        ----------
        param path_to_data_directory: list,
            The first index of the list represents the path to the nomuinal data folder, and the second index to the adverse data folder. 
        param anomaly: int or list, Optional,
            Specify the anomalies that need to be save, important if multiple anomalies need to be saved.
            The default value is 1
        create_nominal_files : bool, optional
            True will create/update the nominal folder directory. The default is True.
        create_adverse_files : bool, optional
            True will create/update the adverse folder. The default is True.
        **kw : str
            can be used to change default names for the directories by using the  nominal_directory and adverse_directory inputs.

        Returns
        -------
        None.

        """
        counter_nom = 0
        counter_ab = 0
        if type(anomaly) != list:
            anomaly = list([anomaly])

        if not os.path.exists(path_to_data_directory):
            os.makedirs(path_to_data_directory)

        path_nom = os.path.join(path_to_data_directory,
                                kw.pop("nominal_directory","nominal_events"))
        if not os.path.exists(path_nom):
            os.makedirs(path_nom)

        path_adv = os.path.join(path_to_data_directory,
                        kw.pop("adverse_directory","adverse_events"))
        if not os.path.exists(path_adv):
            os.makedirs(path_adv)

        if create_nominal_files:
            temp_df = self.df[self.df.Anomaly==0]
            for fl_id in temp_df.flight_id.unique():
                fl = temp_df[temp_df.flight_id==fl_id]
                fl.flight_id = counter_nom
                fl.to_csv(os.path.join(path_nom, f"data_{counter_nom}.csv"))
                counter_nom += 1
        
        if create_adverse_files:
            for i in anomaly:
                temp_df = self.df[self.df.Anomaly==i]
                for fl_id in temp_df.flight_id.unique():
                    fl = temp_df[temp_df.flight_id==fl_id]
                    fl.flight_id = counter_ab
                    fl.to_csv(os.path.join(path_adv, f"data_{counter_ab}.csv"))
                    counter_ab += 1

        # for fl_id in self.df.flight_id.unique():
        #     fl = self.df[self.df.flight_id==fl_id]
        #     if create_nominal_files:
        #         if fl.Anomaly.iloc[0] == 0:
        #             fl.flight_id = counter_nom
        #             fl.to_csv(os.path.join(path_nom, f"data_{counter_nom}.csv"))
        #             counter_nom += 1
        #     if create_adverse_files:
        #         print(fl.Anomaly.iloc[0])
        #         if fl.Anomaly.iloc[0] in anomaly:
        #             print(counter_ab)
        #             fl.flight_id = counter_ab
        #             fl.to_csv(os.path.join(path_adv, f"data_{counter_ab}.csv"))
        #             counter_ab += 1

    def save(self, filename):
        """
        Save data model as a pickle file
        :param filename: str
            Location of directory and name of the file to be saved
        :return:
        """
        if "pkl" not in filename:
            filename = filename + ".pkl"

        if hasattr(self, "fixed_data_path"):
            filename = os.path.join(self.fixed_data_path, filename)
        with open(filename, "wb") as f:
            pkl.dump(self, f)
        print(f"Data Model Saved! (path: {filename})")
            
    def encodeData(self, type, columns=list):
        if type == "label":
            for feature in columns:
                #TODO: add ability to retrieve encoding
                encoder = LabelEncoder()
                self.df[feature] = encoder.fit_transform(self.df[feature])
        elif type =="onehot":
            raise Exception("Not yet implemented")
        else:
            raise Exception("Type must be label or onehot")

    def normalizeData(self, data, data_set = "train", 
                      use_loaded_df=False, 
                      scaling_method ="zscore"):
        """
        Used to normalized tthe data.

        Parameters
        ----------
        data : numpy array, pandas DataFrame
            data to be scaled. Note that the last two columns will not be scaled as it is assumed they correspond 
            to the Anomaly label and to the flight_id.
        data_set : str, optional
            train, validation, and test must be used to specify which data pass. The default is "train".
        use_loaded_df : bool, optional
            if True the dataFrame availeable in dataContainer.df will be normalized . The default is False.
        scaling_method : str, optional
            z score or min max normalization. The default is "zscore".

        Raises
        ------
        NotImplementedError
            Only two scaling methods availeable.

        Returns
        -------
        scaled_data : numpy array
            scaled data is returned.

        """
        if use_loaded_df:
            if scaling_method == "zscore":
                self.scaler = StandardScaler() #MinMaxScaler(**kwargs)
            elif scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                print("use zscore or minmax")
                raise NotImplementedError
            scaled_data = self.scaler.fit_transform(self.df)
            self.df = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)
        else:
            # Make sure that the last cols of kwargs["data"] are anomaly and flight id
            if data_set == "train":
                if scaling_method == "zscore":
                    self.scaler = StandardScaler()
                elif scaling_method == "minmax":
                    self.scaler = MinMaxScaler()
                else:
                    print("use zscore or minmax")
                    raise NotImplementedError
                temp = data.reshape(-1, data.shape[-1])[:, -2:] # flight id and anomaly
                scaled_data = self.scaler.fit_transform(data.reshape(-1,
                                                        data.shape[-1]))
                scaled_data[:, -2:] = temp
                self.trainX = scaled_data.reshape(data.shape[0],
                                                  -1, data.shape[-1])
            else:
                temp = data.reshape(-1, data.shape[-1])[:, -2:] # flight id and anomaly
                scaled_data = self.scaler.transform(data.reshape(-1,
                                                        data.shape[-1]))
                scaled_data[:, -2:] = temp
                if data_set =="validation":
                    self.valX = scaled_data.reshape(data.shape[0],
                                                  -1, data.shape[-1])
                elif data_set == "test":
                    self.testX = scaled_data.reshape(data.shape[0],
                                                  -1, data.shape[-1])
            return scaled_data

    def unnormalizeData(self, data, data2=None, cols=None, concatenate=False):
        """
        Used to un-normalized dataset to get the original values back and create a dataframe.

        Parameters
        ----------
        data : numpy array
            the data to be un-normalized. dataContainer.trainX (and others) can be used here.
        data2 : numpy array, optional
            if the input passed in data doesn't contain all the features that were scaled and the features
            Anomaly and fligh_id, then the anomaly feature should be placed here. The default is None.
        cols : list, optional
            the column names for the dataframe created. The default is None, it will use the column names in dataContainer.header.
        concatenate : bool, optional
            set to True if using data2 input. The default is False.

        Returns
        -------
        pandas DataFrame
            Dataframe containing the the original, non-scaled values for the data.

        """
        if len(data.shape) > 2:
            if concatenate:
                data =  np.concatenate((data, 
                    np.broadcast_to(np.array(data2.flatten())[:, None, None], 
                                    data.shape[:-1] + (1,))), axis = -1)
            data = data.reshape(-1, data.shape[-1])
        if cols is not None:
            cols = cols
        else:
            cols = self.header.drop(columns=["filename"])
        return pd.DataFrame(self.scaler.inverse_transform(data), columns= cols)

    def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        This formulation assumes a time step of 1 will be used
        """
        data= self.df
        feature_names = self.df.columns.values
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (feature_names[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s' % (feature_names[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (feature_names[j], i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        self.df = agg


    def correlated_feature_remover(self, correlation_threshold=0.9,
                                   donotdelete=None, force_dropped=None, **kw):
        """
        Only keep numerical features that have a pearson correlation less than the specified threshold. The first feature of a pair of 
        highly correlated feature is dropped, unless that feature corresponds to a feature that should be deleted according to the
        donotdelete input.

        Parameters
        ----------
        correlation_threshold : float, optional
            DESCRIPTION. The default is 0.9.
        donotdelete : list, optional
            list of features that should not be removed. The default is None.
        force_dropped : list, optional
            list of features that should be removed. The default is None.
        **kw : TYPE
            used to access DataFrame.select_dtypes includes features types.

        Returns
        -------
        None.

        """
        original_n_features = self.df.columns.shape[0]
        data = self.df.select_dtypes(include=kw.pop("include",
                                                    ["float16", "float32", "float64"]))
        corr = data.corr()
        self.correlation_matrix = corr
        n_features =  data.columns.shape[0]
        corr = corr[(abs(corr) > correlation_threshold) & (corr != 1)]

        colsdropped = []
        features_visited = []

        for feature in corr.columns:
            if len(features_visited) == corr.columns.shape[0]:
                break
            cols2drop = corr.columns[
                corr.loc[feature, :].notna()]  # return cols that have no nans (i.e highly correlated ones)
            features_visited.append(feature)
            for col in cols2drop:
                if col != feature:
                    features_visited.append(col)
            if donotdelete is not None:
                if any([True if feat in donotdelete else False for feat in cols2drop.values ]):
                    colsdropped.append(feature)
                [colsdropped.append(i) for i in cols2drop.values if i not in donotdelete]
            else:
                [colsdropped.append(i) for i in cols2drop.values]
            

        if donotdelete is not None:
            # making sure cols not to deleted are not deleted
            [colsdropped.remove(feat) for feat in donotdelete if feat in colsdropped]
            
        # Delete all columns at once
        if force_dropped is not None:
            for feat in force_dropped:
                try:
                    corr.drop(columns = [feat], inplace=True)
                    corr.drop(index = [feat], inplace=True)
                except:
                    pass
        corr.drop(columns=colsdropped, inplace=True)
        corr.drop(index=colsdropped, inplace=True)

        imp_cols = list(corr.columns)
        if "filename" not in imp_cols:
            imp_cols.append("filename")
        else:
            #to keep order consistent
            imp_cols.remove("filename")
            imp_cols.append("filename")

        if "flight_id" not in imp_cols:
            imp_cols.append("flight_id")
        else:
            imp_cols.remove("flight_id")
            imp_cols.append("flight_id")

        if "Anomaly" not in imp_cols:
            imp_cols.append("Anomaly")
        else:
            imp_cols.remove("Anomaly")
            imp_cols.append("Anomaly")

        # DASHLink data specific problem 
        try:
            imp_cols.remove("Date (Day)")
        except:
            pass
        try:
            imp_cols.remove("Date (Month)")
        except:
            pass
        try:
            imp_cols.remove("Date (Year)")
        except:
            pass 
        try:
            imp_cols.remove('GREENWICH MEAN TIME (HOUR)')
        except:
            pass
        try:
            imp_cols.remove('GREENWICH MEAN TIME (MINUTE)')
        except:
            pass
        try:
            imp_cols.remove('GREENWICH MEAN TIME (SECOND)')
        except:
            pass 
        
        
        self.final_features = imp_cols
        self.df_pre_corr_feature_remover = self.df 
        self.df = self.df[imp_cols]
        self.header = self.df.columns
        
       
    def MIL_processing(self, test_ratio=0.3, val_ratio=0, anomaly=1):
        """
        One of the core methods, it processes the dataframe that the data container stores and 
        shape the data such that it appropriate for the MIL format along with splitting the data into train, test, and validation.
        The data is split using scikit-learn train_test_split function, and keep each set stratified.
        Resulting data shape is [Number of flights, Number of time-steps, Number of features]


        Parameters
        ----------
        test_ratio : float, optional 
            size of the test set. The default is 0.3.
        val_ratio : float, optional
            size of the validation set, which is a portion of the test size. The default is 0.
        anomaly : int or list, optional
            anomaly label. The default is 1.

        Raises
        ------
            DESCRIPTION

        Returns
        -------
        None. But results can accessed using dataContainer.trainX, dataContainer.valX, dataContainer.testX, dataContainer.testX, etc.

        """
        assert test_ratio >= 0
        assert val_ratio >= 0
        if type(anomaly) == list:
            nom_flights = int(self.df[self.df.Anomaly==0].flight_id.unique().shape[0])
            adverse_flights = 0
            for idx in anomaly:
                adverse_flights += int(self.df[self.df.Anomaly==idx].flight_id.unique().shape[0])
            n_flights = nom_flights + adverse_flights
        else:
            anomaly = list([anomaly])
            # Flatten and concatenate all flight ids
            n_flights = int(self.df[self.df.Anomaly==0].flight_id.unique().shape[0] + \
                self.df[self.df.Anomaly==anomaly[0]].flight_id.unique().shape[0])

        flight_class = np.ones((n_flights,1)) *10
        split_index = np.arange(0, n_flights,1).reshape(-1,1)
        temp = []
        counter = 0

        # Fill all the nominal flights
        list_ids = self.df[self.df.Anomaly == 0].flight_id.unique().astype(int)
        for id in list_ids:
            fl = self.df[(self.df.flight_id == id) & (self.df.Anomaly == 0)]
            flight_class[counter] = fl.Anomaly.iloc[0]
            # fl = fl.drop(columns=["Anomaly","flight_id"]).values.reshape(1, -1)
            temp.append(fl)  # .drop(columns=["Anomaly","flight_id"]))
            counter = counter + 1
        # Fill all adverse_flights
        for idx in anomaly:
            list_ids = self.df[self.df.Anomaly == idx].flight_id.unique().astype(int)
            for id in list_ids:
                fl = self.df[(self.df.flight_id == id) & (self.df.Anomaly == idx)]
                flight_class[counter] = fl.Anomaly.iloc[0]
                temp.append(fl)
                counter = counter + 1

        # Stacking flights together
        try: 
            data = np.stack(temp, axis=0) # shape [# flights, sample per flight, n_features]
        except ValueError as err:
            if err=="all input arrays must have the same shape":
                print(err)
                print("Make sure the data used is coming from data obtaining from two different folder (nominal_events, abnormal events")
            raise 
        except: 
            raise
        x_train_idx, x_test_idx, y_train, y_test = train_test_split(split_index,flight_class,
                                                   test_size=test_ratio, stratify=flight_class )

        if len(np.unique(y_train) )== 2: # binary
            y_train[np.where(y_train!=0)] = 1
            y_test[np.where(y_test != 0)] = 1

        self.trainX= data[x_train_idx].reshape(len(x_train_idx), fl.shape[0],-1)
        if val_ratio > 0:
            x_test_idx, x_val_idx, y_test, y_val = train_test_split(x_test_idx, y_test,
                                                   test_size=val_ratio, stratify=y_test)
            self.valIndex = x_val_idx
            self.valX = data[x_val_idx].reshape(len(x_val_idx), fl.shape[0],-1)
            self.valY = y_val
            self.using_val_data = True
        else:
            self.using_val_data = False

        self.trainIndex = x_train_idx
        self.testX = data[x_test_idx].reshape(len(x_test_idx), fl.shape[0],-1)
        self.trainY, self.testY = y_train, y_test
        self.testIndex = x_test_idx

    def nom_flights_mean_std(self, mle=True):
        """
        Finds the mean and standard variation for each time-step (temporal mean and standard deviation).         

        Parameters
        ----------
        mle : bool, optional
            if True, the maximum likelihood estimate will be computed for each feature's mean. The default is True.

        Returns
        -------
        None. But results can be accessed using dataContainer.mus, dataContainer.sigmas

        """
        n_pts = self.max_len
        df = self.df
        mus_temporal = np.zeros((n_pts, df.shape[1] - 2))
        sigmas_temporal = np.zeros((n_pts, df.shape[1] - 2))
        nominal_fl = df[df.Anomaly == 0]

        if mle:
            # temporal Gaussians
            for i in range(n_pts):
                for j in range(df.shape[1] - 2):
                    vector = nominal_fl.iloc[i::n_pts, j]  # Make sure you're using unscaled values
                    mus_temporal[i, j], sigmas_temporal[i, j] = norm.fit(vector)
        else:
            #TODO: Implement non-mle way
            raise NotImplementedError
        self.mus = mus_temporal
        self.sigmas = sigmas_temporal
        
    def normalize_resample_data(self, sampling_method=None,**kw):
        """
        Usually used after the MIL_processing method. The data is normalized using the train set data.
        The train set can also then be resampled so that the training data is overfitted or underfitted, if specified. 
        Parameters
        ----------
        sampling_method : str or imblearn function, optional
            The sampling methods that will used. Availeable methods are SMOTE, random oversampler, random undersampler, SMOTETomek. 
            If None, no resampling will be done. The user can also insert any sampling method from the imblearn
            package that uses the fit_sample method. The default is None.
        **kw : 
            Can be used to input parameters for the initiation of a imblearn function

        Raises
        ------
        NotImplemented
            Method is not availeable by default, insert the imblearn function.

        Returns
        -------
        None. But results can be accesed using dataContainer.trainX, etc. (see MIL method).
        Additionaly the resampled results can be accesed via dataContainer.X_res, and dataContainer.y_res

        """
        # Save filenames 
        remove_filename = kw.pop("remove_filename", True)
        if remove_filename:
            filename_pos = np.where("filename"==self.header)[0][0]
            if self.using_val_data:
                filenames = np.concatenate([self.trainX[:, 0, filename_pos],
                                        self.valX[:, 0, filename_pos],
                                        self.testX[:, 0, filename_pos]])
            
            else:
                filenames = np.concatenate([self.trainX[:, 0, filename_pos],
                                        self.testX[:, 0, filename_pos]])
            self.filenames = filenames
            # Remove filenames from X 
            self.trainX = np.delete(self.trainX, filename_pos, axis=2)
            if self.using_val_data:
                self.valX = np.delete(self.valX, filename_pos, axis=2)
            self.testX = np.delete(self.testX, filename_pos, axis=2)
        method = kw.pop("scaling_method","zscore")
        self.normalizeData(use_loaded_df=False, data_set="train", data = self.trainX, scaling_method=method)
        if self.using_val_data:
            self.normalizeData(use_loaded_df=False, data_set="validation",  data = self.valX, scaling_method=method)
        self.normalizeData(use_loaded_df=False, data_set="test", data = self.testX, scaling_method=method)

        # Performing sampling
        if sampling_method is not None:
            if type(sampling_method) == str:
                if sampling_method == "SMOTE":
                    sampler = SMOTE(random_state=3)
                elif sampling_method == "random oversampler":
                    sampler = RandomOverSampler(random_state=3)
                elif sampling_method == "random undersampler":
                    sampler = RandomUnderSampler(random_state=3) # init undersampler
                elif sampling_method == "SMOTETomek":
                    sampler = SMOTETomek(random_state=3)
                else:
                    print("use: SMOTE, random oversampler, random undersampler, SMOTETomek")
                    raise NotImplemented
            else:
                sampler = sampling_method(kw)
        X_train, y_train = self.trainX, self.trainY
        print(f"training size: {X_train.shape}")
        if self.using_val_data:
            print(f"validation size: {self.valX.shape}")
        X_test, y_test = self.testX, self.testY
        print(f"test size: {X_test.shape}")
        
        if sampling_method is not None:
            x = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
            X_res, y_res = sampler.fit_resample(x, y_train)
            X_res = X_res.reshape(-1, X_train.shape[1], X_train.shape[2])
            print(f"Resampled training size: {X_res.shape}\nSizes include two features not used i.e. flight_id and the anonaly type")
            self.x_train = x 
            self.X_res = X_res
            self.y_res = y_res

class myDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]