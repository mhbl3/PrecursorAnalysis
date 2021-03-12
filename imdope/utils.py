import matplotlib.pyplot as plt
import numpy as np
from .dataModel.dataProcessing import *
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
import warnings 
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.decomposition import PCA
import pickle as pkl
from os.path import join
import scipy.integrate as integrate
import time



def window_padding(data_to_pad, full_length, method="interp"):
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
    
    if method == "interp":
        x = np.linspace(0, full_length-1, full_length)
        xp = np.linspace(0, full_length-1, data_to_pad.shape[0]) #might need to be -n instead of 1
        fp = data_to_pad
        return np.interp(x=x, xp=xp, fp=fp)
    elif method =="steps":
        seq_len = int(round(full_length/data_to_pad.shape[0]))
        temp = np.zeros((full_length, 1))
        for n, m in enumerate(range(0, temp.shape[0], seq_len)):
            try:
                temp[m:m+seq_len] = data_to_pad[n]
            except:
                temp[m:m+seq_len] = data_to_pad[-1]
        return temp

def plot_feature_effects(effect, proba_time, flight_id, unormalized_data,
                         columns, mus, sigmas, c=3, ytick_one=True, features_to_show=None,
                         save_path=None):
 """
    Plots the precursor score over for each feature for the specified flight

    Parameters
    ----------
    effect : array of shape [Num flights, Time steps, number of features]
        precursor score over time for each feature.
    proba_time : array of shape [Num flights, Time-steps ]
        Precursor probability over time .
    flight_id : int
        the index of the flight to be plotted.
    unormalized_data : array
        the unormalize un-scaled data for the flight to be plotted.
    columns : list
        list containing feature names.
    mus : array of shape [Num flights, time-steps, num of features]
        the mean temporal values to be used.
    sigmas : array of shape [Num flights, time-steps, num of features]
        the temporal standard deviation values to be used..
    features_to_show : list, optional
        reduced list of features to plot. The default is None.
    save_path : str, optional
        Path to save the plot. The default is None.

    Returns
    -------
    None.

 """  
 # Initializations
 assert type(unormalized_data) == np.ndarray
 if features_to_show is not None:
    assert (len(features_to_show) > 4)
 counter = 0
 width = 4*5.5
 mus_temporal = mus
 upper, lower = boundCreator(mus, sigmas, c)
 if features_to_show is None:
    n_features = mus_temporal.shape[1]
    fl = unormalized_data
 else:
    n_features = len(features_to_show)
    temp = []
    for feat in features_to_show:
        temp_idx = np.where(feat == columns)[0][0]
        temp.append(temp_idx)
    mus_temporal = mus_temporal[:,temp]
    upper = upper[:, temp]
    lower = lower[:, temp]
    columns = columns[temp]
    fl = unormalized_data[:,  temp]
    effect = effect[:, :, temp]

 height = 3.5* int(n_features/4+1)
 fig, ax1 = plt.subplots(int(n_features/4+1), 4, figsize=(width, height))
 fig.tight_layout(pad=6.5)
 l = mus_temporal.shape[0]

 for i in range(0, int(n_features-4)+1):
     for j in range(0, 4):
         if i==0 and j==0:
             if proba_time.shape[1] != mus_temporal.shape[0]:
                 # pad values
                 r_proba_time = window_padding(proba_time[flight_id, :].flatten(), mus_temporal.shape[0])
             else:
                 r_proba_time = proba_time[flight_id, :]
             ax1[i,j].plot(r_proba_time, "r")
             ax1[i,j].set_ylabel("Probability")
             ax1[i,j].set_title("Precursor Score")
             ax1[i,j].grid(True)
             ax1[i,j].set_xlabel("Distance to Event (nm)")
            #  ax1[i,j].set_yticks(np.arange(0, 1.1, 0.1))
             x = np.arange(20 , -0.25, -0.25)
             ax1[i,j].set_xticks(range(0, l, 10))
             ax1[i,j].set_xticklabels(x[::10])
             continue
         if counter == n_features:
             break
         effect_value = effect[flight_id, :, counter]
         # In the case the window used does not match the flight length
         if effect_value.shape[0] != mus_temporal.shape[0]:
             # pad values
             effect_value = window_padding(effect_value.flatten(),
                                           mus_temporal.shape[0])
         else:
             effect_value = effect[flight_id, :, counter]
         ax1[i,j].plot(effect_value, "r")
         # ax1[i,j].legend(data_model.df.columns[counter].values)#, loc=4)
         ax1[i,j].set_ylabel("Feature Effect")
         ax2 = ax1[i,j].twinx()
         ax2.plot(fl[:, counter], "--")
         ax2.plot(mus_temporal[:, counter], "--k")
         ax2.fill_between(range(mus_temporal.shape[0]), lower[:, counter], upper[:, counter],
                    color='b', alpha=0.25)
         ax2.set_ylabel("Feature values")
         ax1[i,j].set_title(f"Feature: {columns[counter]}")
         ax1[i,j].set_xlabel("Distance to Event (nm)")
         ax1[i,j].set_ylabel("Score")
         ax1[i,j].grid(True)
         if ytick_one == True:
            ax1[i,j].set_yticks(np.arange(0, 1.1, 0.1))
         else:
            ax1[i,j].set_yticks(np.arange(0, .6, 0.1))
         x = np.arange(20 , -0.25, -0.25)
         ax1[i,j].set_xticks(range(0, l, 10))
         ax1[i,j].set_xticklabels(x[::10])
         counter += 1

 if save_path is not None:
     plt.savefig(save_path, dpi=600)
     

def plot_feature_importance(f_weights, columns, 
                            order="descending", save_path=None):
  plt.figure(figsize=(30,30))
  if order=="descending":
    idx = np.argsort(abs(f_weights))
  elif order =="ascending":
    idx = np.argsort(abs(f_weights))[::-1]
  sorted_columns = columns[idx]
  plt.barh(range(len(idx)), f_weights[idx])
  plt.grid(True)
  plt.yticks(range(len(columns)), sorted_columns)
  for i,j in enumerate(f_weights[idx]):
    if j >= 0:
      c = 0.02
    else:
      c = - 0.02
    plt.text(x = j+c, y= i, s=str(round(j,4)) )
  plt.xlim(min(f_weights)-0.05, max(f_weights)+0.05)

  if save_path is not None:
    plt.savefig(save_path+"//feature_ranking.pdf")


def plot_feature_effects_no_effects(proba_time, flight_id, unormalized_data,
                         columns, mus, sigmas, c=3, ytick_one=True, show_precursor_range=False, features_to_show=None,
                         save_path=None, **kw):
 """
    Plots the precursor score over for each feature for the specified flight

    Parameters
    ----------
    proba_time : array of shape [Num flights, Time-steps ]
        Precursor probability over time .
    flight_id : int
        the index of the flight to be plotted.
    unormalized_data : array
        the unormalize un-scaled data for the flight to be plotted.
    columns : list
        list containing feature names.
    mus : array of shape [Num flights, time-steps, num of features]
        the mean temporal values to be used.
    sigmas : array of shape [Num flights, time-steps, num of features]
        the temporal standard deviation values to be used..
    features_to_show : list, optional
        reduced list of features to plot. The default is None.
    save_path : str, optional
        Path to save the plot. The default is None.

    Returns
    -------
    None.

 """  
 # Initializations
 assert type(unormalized_data) == np.ndarray
 ticks_on = kw.get("ticks_on", True)
 if features_to_show is not None:
    assert (len(features_to_show) > 4)
 counter = 0
 width = 4*5.5
 mus_temporal = mus
 upper, lower = boundCreator(mus, sigmas, c)
 if features_to_show is None:
    n_features = mus_temporal.shape[1]
    fl = unormalized_data
 else:
    n_features = len(features_to_show)
    temp = []
    for feat in features_to_show:
        temp_idx = np.where(feat == columns)[0][0]
        temp.append(temp_idx)
    mus_temporal = mus_temporal[:,temp]
    upper = upper[:, temp]
    lower = lower[:, temp]
    columns = columns[temp]
    fl = unormalized_data[:,  temp]

 height = 3.5* int(n_features/4+1)
 fig, ax1 = plt.subplots(int(n_features/4+1), 4, figsize=(width, height))
 fig.tight_layout(pad=6.5)
 l = mus_temporal.shape[0]
 flag = False

 for i in range(0, int(n_features-4)+1):
     for j in range(0, 4):
         if i==0 and j==0:
             if proba_time.shape[1] != mus_temporal.shape[0]:
                 if proba_time.shape[-1] == 1:
                    # pad values
                    r_proba_time = window_padding(proba_time[flight_id, :].flatten(), mus_temporal.shape[0])
                 else:
                    flag =True
                    r_proba_time = np.zeros(( proba_time.shape[-1], mus_temporal.shape[0]))
                    for prob in range( proba_time.shape[-1]):
                        r_proba_time[prob, :] = window_padding(proba_time[flight_id,: ,prob].flatten(), mus_temporal.shape[0])
             else:
                 r_proba_time = proba_time[flight_id, :]
             if not flag:
                ax1[i,j].plot(r_proba_time, "r")
             else:
                for prob in range( proba_time.shape[-1]):
                    if prob == 0:
                      ax1[i,j].plot(r_proba_time[prob, :],"r")
                    else:
                      ax1[i,j].plot(r_proba_time[prob, :])
             ax1[i,j].set_ylabel("Probability")
             ax1[i,j].set_title("Precursor Score")
             ax1[i,j].grid(True)
             ax1[i,j].set_xlabel("Distance to Event (nm)")
            #  ax1[i,j].set_yticks(np.arange(0, 1.1, 0.1))
             if ticks_on:
                 x = np.arange(20 , -0.25, -0.25)
                 ax1[i,j].set_xticks(range(0, l, 10))
                 ax1[i,j].set_xticklabels(x[::10])
             if show_precursor_range:
                if len(r_proba_time.shape) > 1:
                    mask_idx = np.where(r_proba_time[0,:] > 0.5)[0]
                else:
                    mask_idx = np.where(r_proba_time > 0.5)[0]
                diff_time_step_masks = np.diff(mask_idx)
                where_jump = np.where(diff_time_step_masks > 1)[0] +1
                ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')

             continue
         if counter == n_features:
             break
         

         ax1[i,j].plot(fl[:, counter], "--")
         ax1[i,j].plot(mus_temporal[:, counter], "--k")
         ax1[i,j].fill_between(range(mus_temporal.shape[0]), lower[:, counter], upper[:, counter],
                    color='b', alpha=0.25)
         ax1[i,j].set_ylabel("Feature values")
         ax1[i,j].set_title(f"Feature: {columns[counter]}")
         ax1[i,j].set_xlabel("Distance to Event (nm)")
         ax1[i,j].grid(True)

         if ticks_on:
             x = np.arange(20 , -0.25, -0.25)
             ax1[i,j].set_xticks(range(0, l, 10))
             ax1[i,j].set_xticklabels(x[::10])
         counter += 1
         if show_precursor_range:
            if len(r_proba_time.shape) > 1:
                mask_idx = np.where(r_proba_time[0,:] > 0.5)[0]
            else:
                mask_idx = np.where(r_proba_time > 0.5)[0]
            diff_time_step_masks = np.diff(mask_idx)
            where_jump = np.where(diff_time_step_masks > 1)[0] +1
            ax1[i,j].axvspan(mask_idx[0], mask_idx[-1], alpha=0.3, color='grey')

 if save_path is not None:
     plt.savefig(save_path, dpi=600)


def plot_features_temporal_dist(unormalized_data,
                                columns, mus: list, sigmas: list, c=3,
                                features_to_show=None,
                                save_path=None, **kw):
    # Initializations
    assert (type(unormalized_data) == np.ndarray) or (unormalized_data == None)
    ticks_on = kw.get("ticks_on", True)
    if features_to_show is not None:
        assert (len(features_to_show) > 4), "Feature to show must be greater than 4"

    width = 4 * 5.5
    mu_colors = kw.pop("mu_colors", ["b", "r", "g"])
    fill_colors = kw.pop("fill_colors", ["b", "r", "g"])
    for iteration, (mus_temporal, sigma) in enumerate(zip(mus, sigmas)):
        counter = 0
        upper, lower = boundCreator(mus_temporal, sigma, c)
        if features_to_show is None:
            n_features = mus_temporal.shape[1]
            if unormalized_data is not None:
                fl = unormalized_data
        else:
            if iteration == 0:
                n_features = len(features_to_show)
                temp = []
                for feat in features_to_show:
                    temp_idx = np.where(feat == columns)[0][0]
                    temp.append(temp_idx)
            mus_temporal = mus_temporal[:, temp]
            upper = upper[:, temp]
            lower = lower[:, temp]
            if iteration == 0:
                columns = columns[temp]
            if unormalized_data is not None:
                fl = unormalized_data[:, temp]
        if iteration == 0:
            height = 3.5 * int(n_features / 4 + 1)
            fig, ax1 = plt.subplots(int(n_features / 4 + 1), 4, figsize=(width, height))
            fig.tight_layout(pad=6.5)
            l = mus_temporal.shape[0]

        for i in range(0, int(n_features - 4)):  # +1?
            for j in range(0, 4):
                if counter == n_features:
                    break
                if unormalized_data is not None:
                    ax1[i, j].plot(fl[:, counter], "--k")
                ax1[i, j].plot(mus_temporal[:, counter], "--{}".format(mu_colors[iteration]),
                               label="Anomaly: {}".format(iteration))
                ax1[i, j].fill_between(range(mus_temporal.shape[0]), lower[:, counter], upper[:, counter],
                                       color="{}".format(fill_colors[iteration]), alpha=0.25)
                ax1[i, j].set_ylabel("Feature values")
                ax1[i, j].set_title(f"Feature: {columns[counter]}")
                ax1[i, j].set_xlabel("Distance to Event (nm)")
                ax1[i, j].grid(True)
                ax1[i, j].legend()

                if ticks_on:
                    x = np.arange(20, -0.25, -0.25)
                    ax1[i, j].set_xticks(range(0, l, 10))
                    ax1[i, j].set_xticklabels(x[::10])
                counter += 1

    if save_path is not None:
        plt.savefig(save_path, dpi=600)

def boundCreator(mus, sigmas, c=3):
    """
    Creates interval around the mean

    Parameters
    ----------
    mus : array of shape [Num flights, time-steps, num of features]
        the mean temporal values to be used.
    sigmas : array of shape [Num flights, time-steps, num of features]
        the temporal standard deviation values to be used.
    c : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    upper : array
        upper bound.
    lower : array
        lower bound.

    """
    upper = mus + c*sigmas
    lower = mus - c*sigmas
    return upper,lower

def matcher(data_no_id, data_id, original_df, return_flight_id = False):
    """
    Used to retrieve a correspond flight to a scaled data 

    Parameters
    ----------
    data_no_id : array
        scaled data with no flight _id.
    data_id : array 
        scaled data with flight_id.
    original_df : pd.DataFrame
        original data frame with the un-scaled features and flight_id.
    return_flight_id : bool, optional
        If true only the flight_id will be returned. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    idx = np.where((data_no_id == data_id[:, :, :-2]).all(axis=1).all(axis=1))[0][0]
    flight_id = data_id[idx, 0, -1]
    anomaly = data_id[idx, 0, -2]
    fl = original_df[(original_df.flight_id == flight_id) & (original_df.Anomaly == anomaly)]
    if return_flight_id:
        return flight_id
    else:
        return fl

def history_accurracy_plotter(hist, acc, v_hist=None, v_acc=None, save_path=None):
  fig, ax = plt.subplots(1, figsize=(15,15))
  ax.plot(hist, label="loss")
  if v_hist is not None:
      plt.plot(v_hist, label="val_loss")
  ax2 = ax.twinx()
  ax2.plot(acc, "r", label="acc")
  if v_acc is not None:
      plt.plot(v_acc, "g", label="val_acc")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  ax2.set_ylabel("Balanced Accuracy")
  ax.legend(loc=2); ax2.legend(loc=9)

  if save_path is not None:
    plt.savefig(save_path+"//loss_acc.pdf")

#TODO: Can this function be faster?
def dataResampling(list_df, distance_away=20, n_miles_increment=0.25, adjust_DF=True,
                   limiting_feature="ALTITUDE", handle_categorical=False,
                   limiting_value=1000):
    """
    Used to resample flight data such that all flights used have the same length. Can also specify the cutoff for the data
    (i.e. forecasting vs classification problem)

    Parameters
    ----------
    list_df : list
        list containing pd.DataFrames.
    distance_away : float, optional
        The distance away from the cutoff point The data from that distance to the cutoff point will be returned. The default is 20.
    n_miles_increment : float, optional
        increments from distance away to cutoff point. The default is 0.25.
    adjust_DF : bool, optional
        Some flights do not work well with this function (i.e. cutoff is missing, etc.). If True
        they are automatically deleted. The default is True.
    limiting_feature : str, optional
        feature that is used to create the cutoff. The default is "ALTITUDE".
    limiting_value : str, optional
        value for the limiting feature. The default is 1000.

    Returns
    -------
    list_df : list
        list containing resampled/filtered flights. Must be concatenated all together later.
    not_working_flights : list
        index of flights in original list for which the function did not work.

    """
    assert type(list_df) == list
    not_working_flights = []
    for i, df in enumerate(tqdm(list_df)):
        if limiting_feature is not None:
            df = df[df[limiting_feature] > limiting_value]
        try:
            distance = np.arange(df.DISTANCE_TO_LANDING.iloc[0],
                                 df.DISTANCE_TO_LANDING.iloc[-1],
                                 n_miles_increment)  # Quarter mi increments
            n_pts = len(distance)
            idx = df.index
            temp = pd.DataFrame()
            cols = df.columns
            cat_cols = []
            float_cols = df.select_dtypes(include=["float16",
                                                   "float32",
                                                   "float64"]).columns  # Assumes numerical data is of type

            # Placing the resampled distance inside the dataframe
            for col in cols:
                if col == "DISTANCE_TO_LANDING":
                    temp[col] = distance
                else:
                    # Interpolate other features
                    if col in float_cols:
                        lowerlim, upperlim = idx[0], idx[-1]
                        x = np.linspace(lowerlim, upperlim, n_pts)
                        fp = df[col].values
                        xp = idx
                        resampled_data = np.interp(x=x, xp=xp, fp=fp)
                        temp[col] = resampled_data
                    else:
                        if not handle_categorical:
                            temp[col] = df[col].iloc[0]
                            if i == 0:
                                warnings.warn("This function does not work on categorical variables")
                        else:
                            cat_cols.append(cols)

            # inversing distance to make it the distance left and setting a zero
            temp["DISTANCE_TO_LANDING"] = abs(
                temp.DISTANCE_TO_LANDING - temp.DISTANCE_TO_LANDING.iloc[-1])  # distance remaining
            # Filtering
            temp = temp[temp.DISTANCE_TO_LANDING <= distance_away]  # last 20 mi
            # Looks min distance to point of interest and use the value there for the categorical variable
            if handle_categorical:
                for col in cat_cols:
                    tmp_list_cat = []
                    distance_pts = np.arange(distance_away, 0 - n_miles_increment, -n_miles_increment)
                    df_dist_pts = pd.DataFrame(data={"myvals": distance_pts})
                    series_of_diff = df_dist_pts.myvals.apply(lambda x: abs(x - df.DISTANCE_TO_LANDING.values))
                    array_diff = np.concatenate(series_of_diff.values).reshape(-1, len(distance_pts))
                    indices = np.argmin(array_diff, axis=0)
                    # for distance_pt in distance_pts:
                    #     idx_min_dist = np.argmin(abs(df.DISTANCE_TO_LANDING.values-distance_pt))
                    #     tmp_list_cat.append(df[col].iloc[idx_min_dist])
                    temp[col] = df.reset_index(drop=True).loc[indices, col].values  # np.asarray(tmp_list_cat)

            list_df[i] = temp
        except KeyboardInterrupt:
            raise
        except Exception as err:
            # Handling flights that had an issue
            print(err)
            print(f"Issue at index: {i}")
            list_df[i] = np.nan
            not_working_flights.append(i)
    if adjust_DF:
        # Adjusting the DF list because some flights might have not worked
        for j, i in enumerate(not_working_flights):
            list_df.pop(i - j)

    return list_df, not_working_flights


def dist_plotter(df, dtype="numerical", anomaly=1, show_hist_numerical=False):
    """
    Plots the distribution for each feature showing the nomimal vs adverse flights

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame will all data including nominal and adverse flights.
    dtype : str, optional
        numerical or category can be used. The default is "numerical".
    anomaly : int, optional
        The anomaly index to be plotted. The default is 1.
    show_hist_numerical : bool, optional
        Show histogram instead of density plot. The default is False.

    Returns
    -------
    None.

    """
    temp_nom = df[(df.Anomaly == 0)]
    counter = 0

    if dtype == "numerical":
        temp_nom = temp_nom.select_dtypes(include=["float32", "float64"])
        n_features = temp_nom.shape[1]
        temp_adv = df[(df.Anomaly == anomaly)]
        temp_adv = temp_adv.select_dtypes(include=["float32", "float64"])

        width = 4 * 5.5
        height = 3.5 * int(n_features / 4 + 1)
        fig, axis = plt.subplots(int(n_features / 4 + 1), 4, figsize=(width, height))
        fig.tight_layout(pad=6.5)
        for i in range(0, int(n_features / 4) + 1):
            for j in range(0, 4):
                if counter == n_features:
                    break
                if show_hist_numerical:
                    sns.distplot(temp_nom.iloc[:, counter], ax=axis[i, j], label="Nominal",
                                 kde=False, hist=True, norm_hist=True)
                    sns.distplot(temp_adv.iloc[:, counter], ax=axis[i, j], label="Adverse",
                                 kde=False, hist=True, norm_hist=True)
                else:
                    sns.distplot(temp_nom.iloc[:, counter], ax=axis[i, j], label="Nominal")
                    sns.distplot(temp_adv.iloc[:, counter], ax=axis[i, j], label="Adverse")
                axis[i, j].set_ylabel("Density")
                axis[i, j].set_title(f"Feature: {temp_adv.columns[counter]}")
                axis[i, j].grid(True)
                axis[i, j].legend()
                counter += 1
    elif dtype == "category":
        temp_nom = temp_nom.select_dtypes(include=[dtype])
        n_features = temp_nom.shape[1]
        temp_adv = df[(df.Anomaly == anomaly)]
        temp_adv = temp_adv.select_dtypes(include=[dtype])

        width = 4 * 5.5
        height = 3.5 * int(n_features / 4 + 1)
        fig, axis = plt.subplots(int(n_features / 4 + 1), 4, figsize=(width, height))
        fig.tight_layout(pad=6.5)
        for i in range(0, int(n_features / 4) + 1):
            for j in range(0, 4):
                if counter == n_features:
                    break
                sns.distplot(temp_nom.iloc[:, counter], ax=axis[i, j], label="Nominal", kde=False, hist=True)
                sns.distplot(temp_adv.iloc[:, counter], ax=axis[i, j], label="Adverse", kde=False, hist=True)
                axis[i, j].set_ylabel("Count")
                axis[i, j].set_title(f"Feature: {temp_adv.columns[counter]}")
                axis[i, j].grid(True)
                axis[i, j].legend()
                counter += 1

def layer_heatmap(model, layer, flight_data, feature, feature_list=None,
                figsize=None, annot=False, compute_ranking_across_flights= False,
                  use_max_pool=False, threshold=0.1):
  """
    
    
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    layer : TYPE
        DESCRIPTION.
    flight_data : TYPE
        DESCRIPTION.
    feature : TYPE
        DESCRIPTION.
    feature_list : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is None.
    annot : TYPE, optional
        DESCRIPTION. The default is False.
    compute_ranking_across_flights : TYPE, optional
        DESCRIPTION. The default is False.
    use_max_pool : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.1.
    
    Returns
    -------
    None.
    
  """
  if feature_list is not None:
      feature = np.where(feature==feature_list)[0][0]
  if len(flight_data.shape)==2:
      x = flight_data.reshape(1, flight_data.shape[0], flight_data.shape[1])[:, : , :-2]
      x_tensor = torch.tensor(x).permute(0, 2, 1)[:, feature, :].view(1, 1, -1)
      print(f"Model Prediction: {model.predict(x)[0]}")
  else:
      x = flight_data[:, : , :-2]
      x_tensor = torch.tensor(x).permute(0, 2, 1)[:, feature, :].view(flight_data.shape[0], 1, -1)

  if layer == 1:
      layer =  model.cnns1[feature].double()
      layer_out = layer(x_tensor)
      layer.float()
  elif layer == 2:
      lay1 =  model.cnns1[feature].double()
      layer_out = lay1(x_tensor)
      lay1.float()
      lay2 =  model.cnns2[feature].double()
      layer_out = lay2(layer_out)
      lay2.float()
  elif layer == 3 : 
      lay1 =  model.cnns1[feature].double()
      layer_out = lay1(x_tensor)
      lay1.float()
      lay2 =  model.cnns2[feature].double()
      layer_out = lay2(layer_out)
      lay2.float()
      lay3 =  model.cnns3[feature].double()
      layer_out = lay3(layer_out)
      lay3.float()
  elif layer == 4:
      lay1 =  model.cnns1[feature].double()
      layer_out = lay1(x_tensor)
      lay1.float()
      out1 = layer_out.permute(0, 2, 1) 
      out1 = out1.detach().numpy().reshape(-1, out1.shape[2])
      if len(flight_data.shape) != 2:
          out1 = out1.reshape(flight_data.shape[0], -1, out1.shape[1]).mean(axis=0)
      
      lay2 =  model.cnns2[feature].double()
      layer_out = lay2(layer_out)
      lay2.float()
      out2 = layer_out.permute(0, 2, 1) 
      out2 = out2.detach().numpy().reshape(-1, out2.shape[2])
      if len(flight_data.shape) != 2:
          out2 = out2.reshape(flight_data.shape[0], -1, out2.shape[1]).mean(axis=0)
      
      lay3 =  model.cnns3[feature].double()
      layer_out = lay3(layer_out)
      lay3.float()
      out3 = layer_out.permute(0, 2, 1) 
      out3 = out3.detach().numpy().reshape(-1, out3.shape[2])
      if len(flight_data.shape) != 2:
          out3 = out3.reshape(flight_data.shape[0], -1, out3.shape[1]).mean(axis=0)
      

  
  print("CNN Head:")
  if layer==4:
      print("Multiple Layers")
  else:
      print(layer)
  if figsize is None:
      figsize=(13,10)
  if layer != 4:
      layer_out = layer_out.permute(0, 2, 1) 
      layer_out = layer_out.detach().numpy().reshape(-1, layer_out.shape[2])
      layer_out = layer_out.reshape(flight_data.shape[0], -1, layer_out.shape[1]).mean(axis=0)
      if use_max_pool:
          # Max pool across time 
          layer_max= layer_out.max(axis=0)
          for i in range(layer_out.shape[1]):
              layer_out[:,i] = layer_max[i]
          n_active_layers = len(np.where(layer_max > threshold)[0])
      fig, ax  = plt.subplots(1, figsize=figsize)
      sns.heatmap(layer_out, ax= ax, annot=annot)
      ax.set_xlabel("Filter"); ax.set_ylabel("Time-Step")
  else:
      fig, ax  = plt.subplots(3, figsize=figsize)
      sns.heatmap(out1, ax= ax[0], annot=annot)
      sns.heatmap(out2, ax= ax[1], annot=annot)
      sns.heatmap(out3, ax= ax[2], annot=annot)
      ax[0].set_xlabel("Filter"); ax[0].set_ylabel("Time-Step")
      ax[1].set_xlabel("Filter"); ax[1].set_ylabel("Time-Step")
      ax[2].set_xlabel("Filter"); ax[2].set_ylabel("Time-Step")

  plt.show()
  model.show_feature_importance(feature_list, plot=False)
#     print(model.sorted_features)
  if compute_ranking_across_flights:
      score_tracker = np.zeros((flight_data.shape[0], len(feature_list)-2)) # feature list includes flight_id and Anomaly
      sorted_feat = [feat for feat, score in sorted(zip(model.sorted_features, model.sorted_features_values), 
                                                      key= lambda x: x[0].lower())]
      for i in range(flight_data.shape[0]):
          model.show_feature_importance(feature_list, flight_id=i, plot=False)
          sorted_score = [score for _, score in sorted(zip(model.sorted_features, model.sorted_features_values), 
                                                      key= lambda x: x[0].lower())]
          score_tracker[i, :] = sorted_score
      score_tracker = list(np.nanmean(score_tracker, axis=0))
      feat = [feat for feat, score in sorted(zip(sorted_feat, score_tracker), 
                                                      key= lambda x: x[1])][::-1]
      sorted_val = np.asarray([score for feat, score in sorted(zip(sorted_feat, score_tracker), 
                                                      key= lambda x: x[1])][::-1])
      sorted_val = np.round(sorted_val/sorted_val.sum(),3)
      print([x for x in zip(feat, sorted_val)])
  
  if use_max_pool:
      return [x for x in zip(feat, sorted_val)], n_active_layers
  else:
      return [x for x in zip(feat, sorted_val)]


def create_layer_heatmap(model, layer, feature, feature_list,
                  figsize=None, annot=False,
                  use_max_pool=False, threshold=0.1):
    """


      Parameters
      ----------
      model : TYPE
          DESCRIPTION.
      layer : TYPE
          DESCRIPTION.
      flight_data : TYPE
          DESCRIPTION.
      feature : TYPE
          DESCRIPTION.
      feature_list : TYPE, optional
          DESCRIPTION. The default is None.
      figsize : TYPE, optional
          DESCRIPTION. The default is None.
      annot : TYPE, optional
          DESCRIPTION. The default is False.
      compute_ranking_across_flights : TYPE, optional
          DESCRIPTION. The default is False.
      use_max_pool : TYPE, optional
          DESCRIPTION. The default is False.
      threshold : TYPE, optional
          DESCRIPTION. The default is 0.1.

      Returns
      -------
      None.

    """
    conv_out = model.convolution_ouputs
    feature_index = np.where(feature == feature_list)[0][0]
    conv_out_feat_interest = conv_out[feature_index]
    if layer <= 2:
        conv_out_feat_interest_layer = np.asarray(conv_out_feat_interest[layer])
        layer_out = np.average(conv_out_feat_interest_layer, axis=0) # average across all flights
        if figsize is None:
            figsize = (13, 10)
        if use_max_pool:
            # Max pool across time
            layer_max= layer_out.max(axis=0)
            for i in range(layer_out.shape[1]):
                layer_out[:,i] = layer_max[i]
            n_active_layers = len(np.where(layer_max > threshold)[0])
            print(f"Number active layer: {n_active_layers}")
        fig, ax = plt.subplots(1, figsize=figsize)
        sns.heatmap(layer_out, ax=ax, annot=annot)
        ax.set_title(feature)
        ax.set_xlabel("Filter");
        ax.set_ylabel("Time-Step")
    else:
        for i in range(len(conv_out_feat_interest)):
            conv_out_feat_interest_layer = np.asarray(conv_out_feat_interest[i])
            layer_out = np.average(conv_out_feat_interest_layer, axis=0)  # average across all flights
            if figsize is None:
                figsize = (13, 10)
            if use_max_pool:
                # Max pool across time
                layer_max= layer_out.max(axis=0)
                for i in range(layer_out.shape[1]):
                    layer_out[:,i] = layer_max[i]
                n_active_layers = len(np.where(layer_max > threshold)[0])
                print(f"Number active layer: {n_active_layers}")
            fig, ax = plt.subplots(1, figsize=figsize)
            sns.heatmap(layer_out, ax=ax, annot=annot)
            ax.set_title(feature)
            ax.set_xlabel("Filter")
            ax.set_ylabel("Time-Step")
    plt.show()

def compute_ranking_across_flights(model, feature_list, cumulative_score=True, rounding=3, **kw):
    n_flights = model.proba_time.shape[0]
    model.show_feature_importance(feature_list, plot=False, **kw)
    p=1
    while np.isnan(model.sorted_features_values).any():
        model.show_feature_importance(feature_list, flight_id =p, plot=False, **kw)
        p+=1
    score_tracker = np.zeros((int(n_flights), len(feature_list) - 2))
    
    sorted_feat = [feat for feat, score in sorted(zip(model.sorted_features, model.sorted_features_values), 
                                                      key= lambda x: x[0].lower())]
    non_continuous_score_ranking = []
    list_flight_multiple_scores = []
    list_flight_unique_scores = []
    list_first_index = [] 
    for i in range(n_flights):
        model.show_feature_importance(feature_list, flight_id=i, plot=False, **kw)
        if not np.isnan(model.sorted_features_values).any():
            sorted_features_values = model.sorted_features_values
            sorted_score = [score for _, score in sorted(zip(model.sorted_features, sorted_features_values),
                                                     key=lambda x: x[0].lower())]
            score_tracker[i, :] = sorted_score
            list_flight_unique_scores.append(i)
            list_first_index.append(model.first_flag_index)
        else:
            for j in range(len(model.list_sorted_features_values)):
                sorted_features_values = model.list_sorted_features_values[j]
                sorted_features = model.list_sorted_features[j]
                sorted_score = [score for _, score in sorted(zip(sorted_features, sorted_features_values),
                                                     key=lambda x: x[0].lower())]
                non_continuous_score_ranking.append(sorted_score)
                list_flight_multiple_scores.append(i)
                list_first_index.append(model.first_flag_index)
    if len(non_continuous_score_ranking) > 0:
        score_tracker = np.concatenate((score_tracker, np.asarray(non_continuous_score_ranking)))
    score_tracker_avg = list(np.nanmean(score_tracker, axis=0))
    feat = [feat for feat, score in sorted(zip(sorted_feat, score_tracker_avg),
                                           key=lambda x: x[1])][::-1]
    sorted_val = np.asarray([score for feat, score in sorted(zip(sorted_feat, score_tracker_avg),
                                                             key=lambda x: x[1])][::-1])
    if cumulative_score:
        sorted_val = np.round(sorted_val / sorted_val.sum(), rounding)

    return [x for x in zip(feat, np.round(sorted_val, rounding))], score_tracker, sorted_feat, list_flight_multiple_scores, list_flight_unique_scores, list_first_index

    
def pca_plot(X, y, data_time_series=True,
             label='Classes', **kw):    
    """
    Plot pca approximation of the data

    Parameters
    ----------
    X : array
        2D or 3D array containing the data to be reduced.
    y : int
        labels.
    data_time_series : bool, optional
        True if using a 3D array assuming the second dimension represents time. The default is True.
    label : str, optional
        title name. The default is 'Classes'.
    **kw : TYPE
        DESCRIPTION.

    Returns
    -------
    x_pca : TYPE
        DESCRIPTION.

    """             
    pca = PCA(n_components=2)
    return_reduced_data = kw.pop("return_reduced_data", False)
    if data_time_series:
        x_pca =  pca.fit_transform(X[:, :, :-2].reshape(-1, X.shape[2]-2))
    else:
        x_pca =  pca.fit_transform(X)
    if kw.pop("verbose", 0) > 0:
        print("Variance of each component: {}".format(pca.explained_variance_ratio_))
        print("Total Variance Explained: {}".format(round(sum(list(pca.explained_variance_ratio_))*100,2)))
    plot_2d_space(x_pca, y.flatten(), **kw, label=label)
    
    if return_reduced_data:
        return x_pca
    
def plot_2d_space(X, y, label='Classes', **kw):   
#     colors = ['#1F77B4', '#FF7F0E']
#     markers = ['o', 's']
    for i in np.unique(y):
        temp_x = X[np.where(y==i)[0], :]
        plt.scatter(
            temp_x[:,0], temp_x[:,1],
             **kw,
             label=str(i) )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
def get_flight_filename_from_flight_id(X, y, dataframe, path_to_meta=None):
    mini_xs = X[:,0, -1]
    lst_filename = []
    for i in range(mini_xs.shape[0]):
        fl_id = mini_xs[i]
        event= y[i, 0]
        filename = np.unique(dataframe[(dataframe.flight_id==fl_id) & (dataframe.Anomaly==event)].filename)[0]
        lst_filename.append(filename)
    if path_to_meta is not None:
        meta = pd.read_csv(path_to_meta, index_col="filename")
        meta = meta.iloc[1:, :]
        meta_interest = meta.loc[lst_filename, :]
    return lst_filename, meta_interest

def get_original_flight_from_flight_id(X, y, dataframe, index=None):
    mini_xs = X[:,0, -1]
    lst_flights = []
    for i in range(mini_xs.shape[0]):
        fl_id = mini_xs[i]
        event= y[i, 0]
        flight = dataframe[(dataframe.flight_id==fl_id) & (dataframe.Anomaly==event)]
        if index is not None:
            flight.index = index
        lst_flights.append(flight)
    return pd.concat(lst_flights)

def feature_plotter(df, mus, sigmas, features, anomaly=1, **kw):
    # temp_nom = df[(df.Anomaly == 0)]
    counter = 0
    
    f_idx = [i for i,feat in enumerate(df.columns) if feat in features ]
    n_features = len(f_idx)
    temp_adv = df[(df.Anomaly == anomaly)]
    
    width = 4 * 5.5
    height = 3.5 * int(n_features / 4 + 1)
    fig, axis = plt.subplots(int(n_features / 4 + 1), 4, figsize=(width, height))
    fig.tight_layout(pad=6.5)
    for i in range(0, int(n_features / 4) + 1):
        for j in range(0, 4):
            if counter == n_features:
                break
            axis[i,j].plot(mus[:, f_idx[counter]], "r")
            if i ==0 and j == 0:
                kwargs = dict()
                kwargs["c"] = kw.pop("c", 3)
                kwargs["alpha"] = kw.pop("alpha", 0.2)
                kwargs["lw"] = kw.pop("lw", 1)
            u,l = boundCreator(mus, sigmas, c=kwargs["c"] )
            axis[i, j].plot(u[:, f_idx[counter]], "--r")
            axis[i, j].plot(l[:, f_idx[counter]], "--r")
            temp_adv.iloc[:,  f_idx[counter]].plot(ax = axis[i, j], 
                                           lw=kwargs["lw"], alpha=kwargs["alpha"])
            axis[i, j].set_ylabel("Feature Value")
            axis[i, j].set_title(f"Feature: {temp_adv.columns[f_idx[counter]]}")
            axis[i, j].grid(True)
            counter += 1
            
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def change_df_index(df, path_save=None,index=None, anomaly=0):
    df_reduced = df[df.Anomaly == anomaly]
    if index == None:
        index = np.arange(20, -.25, -.25)
    df_list = []
    for id in df_reduced.flight_id.unique():
        df = df_reduced[df_reduced.flight_id == id]
        df.index = index
        df_list.append(df)
    df_reduced = pd.concat(df_list)
    if path_save is not None:
        df_reduced.to_csv(path_save)
    return df_reduced

def sample_gaussian(m, v, device):
    sample = torch.randn(m.shape, device=device)#.cuda()
    z = m + (v**0.5)*sample
    return z

def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = torch.nn.functional.softplus(h) + 1e-8
    return m, v

def kl_normal(qm, qv, pm, pv, yh):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    #print("log var1", qv)
    return kl

def feature_map_size(W_in, kernel, stride=1, padding=0 ):
    return int(np.floor((W_in+2*padding-(kernel-1)-1)/stride +1))

def anomaly_detection_train_loop(model, X_train, y_train, batch_size_percent=0.10,
                                 optimizer="adam", learning_rate=1e-3, l2 =0, lamda = 100, alpha=1, momentum=0.999,
                                 X_val=None, y_val=None, n_epochs=100, use_stratified_batch_size=True, beta=None,
                                 print_every_iteration=20):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                  lr=learning_rate, weight_decay=l2)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                 lr=learning_rate, weight_decay=l2, momentum=momentum)

    device = model.device
    model.train()
    if beta is None:
        beta = DeterministicWarmup(n=n_epochs, t_max=1)

    if not torch.is_tensor(X_train):
        X_train = torch.Tensor(X_train)
    if not torch.is_tensor(y_train):
        y_train = torch.Tensor(y_train)

    if X_val is not None:
        if not torch.is_tensor(X_val):
            X_val = torch.Tensor(X_val)
        if not torch.is_tensor(y_val):
            y_val = torch.Tensor(y_val)
        data_val = myDataset(X_val, y_val)
    if batch_size_percent > 1:
        batch_size = batch_size_percent
    else:
        batch_size = int(batch_size_percent*X_train.shape[0])
    data_train = myDataset(X_train, y_train)
    if use_stratified_batch_size is False:
        print("Mini-batch strategy: Random sampling")
        dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    else:
        print("Mini-batch strategy: Stratified")
        # get class counts
        weights = []
        for label in torch.unique(y_train):
            count = len(torch.where(y_train == label)[0])
            weights.append(1 / count)
        weights = torch.tensor(weights).to(device)
        samples_weights = weights[y_train.type(torch.LongTensor).to(device)].flatten()
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
        dataloader_train = DataLoader(data_train, batch_size=batch_size, sampler=sampler)

    if X_val is not None:
        dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)
    if print_every_iteration is None:
        print_every_iteration= int(.20*len(X_train))
    try:
        train_loss_epoch_saved = []
        bce_epoch_saved = []
        rec_epoch_saved = []
        for epoch in tqdm(range(n_epochs)):
            train_loss_epoch = 0
            bce_epoch = 0
            rec_epoch = 0

            if X_val is not None:
                val_loss_epoch = 0
                val_bce_epoch = 0
                val_rec_epoch = 0
                val_loss_epoch_saved = []
                val_bce_epoch_saved = []
                val_rec_epoch_saved = []
            for iteration, (batch_x, batch_y) in enumerate(dataloader_train):
                batch_x, batch_y = batch_x.to(device).permute(0, 2, 1), batch_y.to(device)
                loss, _, _, _, rec, kl, ce_lamda, bce = model.loss_function(batch_x, batch_y,
                                                                            beta=next(beta) if type(beta)==DeterministicWarmup else beta,
                                                                            lamda=lamda,
                                                                            alpha=alpha)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                bce_epoch += bce.item()
                rec_epoch += rec.item()

                if iteration % print_every_iteration == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRec Error: {:.6f}\tBCE: {:.6f}'.format(
                        epoch, iteration * len(batch_x), len(dataloader_train.dataset),
                               100. * iteration *len(batch_x) / len(dataloader_train.dataset), loss.item() / len(batch_x), rec.item()/len(batch_x), bce.item()/len(batch_x)))
                if X_val is not None:
                    with torch.no_grad():
                        for val_batch_x, val_batch_y in dataloader_val:
                            val_batch_x, val_batch_y = val_batch_x.to(device).permute(0, 2, 1), val_batch_y.to(device)
                            loss_val, _, _, _, rec_val, _, _, bce_val = model.loss_function(val_batch_x, val_batch_y, beta=beta, lamda=lamda)
                            val_loss_epoch += loss_val.item()
                            val_bce_epoch += bce_val
                            val_rec_epoch += rec_val
                            if iteration % 100 == 0:
                                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tRec Error: {:.4f}\tBCE: {:.4f}'.format(
                                    epoch, iteration * len(val_batch_x), len(dataloader_val.dataset),
                                           100. * iteration / len(dataloader_val), loss_val.item() / len(val_batch_x),
                                            rec_val, bce_val))

            train_loss_epoch_saved.append(train_loss_epoch / len(dataloader_train.dataset))
            bce_epoch_saved.append(bce_epoch / len(dataloader_train.dataset))
            rec_epoch_saved.append(rec_epoch / len(dataloader_train.dataset))

            print('\n====> Train Epoch: {} Average loss: {:.6f} Average_Rec: {:.6f} Average_BCE: {:.6f}'.format(epoch,
                                                                                            train_loss_epoch_saved[epoch],
                                                                                            rec_epoch_saved[epoch],
                                                                                            bce_epoch_saved[epoch]))
            if X_val is not None:
                val_loss_epoch_saved.append(val_loss_epoch / len(dataloader_val.dataset))
                val_bce_epoch_saved.append(val_bce_epoch / len(dataloader_val.dataset))
                val_rec_epoch_saved.append(val_rec_epoch / len(dataloader_val.dataset))
                print('\n====> Validation Epoch: {} Average loss: {:.6f} Average_Rec: {:.6f} Average_BCE: {:.6f}'.format(epoch,
                                                                                                            val_loss_epoch_saved[
                                                                                                                epoch],
                                                                                                            val_rec_epoch_saved[
                                                                                                                epoch],
                                                                                                            val_bce_epoch_saved[
                                                                                                                epoch]
                                                                                                                     ))

    except KeyboardInterrupt:
        print("Returning model up to epoch {}".format(epoch))
    except:
        raise

    if X_val is None:
        return model.eval(), train_loss_epoch_saved, bce_epoch_saved, rec_epoch_saved
    else:
        return model.eval(), train_loss_epoch_saved, bce_epoch_saved, rec_epoch_saved, \
                val_loss_epoch_saved, val_bce_epoch_saved, val_rec_epoch_saved


def batch_prediction(model, x, batch_size=32, **kw):
    y_de = torch.zeros((len(x), 1))
    prediction_dataset = myDataset(x, y_de)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
    del prediction_dataset
    mu_latent, var_latent, predict, x_re = [], [], [], []
    for batch_x, _ in prediction_dataloader:
        (_, _, _), (_, _, _), (_, _, _), \
        (_, _, _), (_, mu_latent_batch, var_latent_batch), _, \
        (_, _, _), (_, _, _), \
        (_, _, _), _, (_, _, _), \
        (_, _, _, _, _, _, _, _), _, _, _, predict_batch = model.lnet(batch_x.to(model.device))
        mu_latent.append(mu_latent_batch)
        var_latent.append(var_latent_batch)
        predict.append(predict_batch)
        # x_re.append(x_re_batch)
    return mu_latent, var_latent, predict#, x_re

def batch_prediction_rec(model, x, batch_size=32, **kw):
    y_de = torch.zeros((len(x), 1))
    prediction_dataset = myDataset(x, y_de)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
    del prediction_dataset
    x_re = []
    for batch_x, _ in prediction_dataloader:
        (_, _, _), (_, _, _), (_, _, _), \
        (_, _, _), (_, _, _), _, \
        (_, _, _), (_, _, _), \
        (_, _, _), _, (_, _, _), \
        (_, _, _, _, _, _, _, _), _, x_re_batch, _, predict_batch = model.lnet(batch_x.to(model.device))
        x_re.append(x_re_batch)
    return x_re

class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t



def create_DF_list(meta_df_events:pd.DataFrame, path_to_tails:str, anomaly_tag:int, LIMIT=None, file_repeat_save=None):
    start = time.time()
    DF_list = []
    missing_data_flight_id = []
    no_file = []
    if (file_repeat_save is not None) and (os.path.exists(file_repeat_save)):
        with open(file_repeat_save, "rb") as f:
            DF_list = pkl.load(f)
        start_index = len(DF_list)-1
    else:
        start_index = 0


    for j, flight in enumerate(tqdm(meta_df_events.index[start_index:]), start_index):
        # get tail
        tail = flight[:3]
        path_to_flight = join(path_to_tails, f"Tail_{tail}" ,  f"Tail_{tail}", flight)
        try:
            with open(path_to_flight, "rb") as f:
                temp = pkl.load(f)
        except:
            if not os.path.exists(path_to_flight):
                no_file.append(path_to_flight)
                path_to_flight = join(path_to_tails, f"Tail_{tail}", flight)
                try:
                    with open(path_to_flight, "rb") as f:
                        temp = pkl.load(f)
                except:
                    raise Exception(f"problem with loading the file, flight: {flight}")
            else:
                raise Exception(f"problem with loading the file, flight: {flight}")
            continue
        temp = temp["data"]
        if any(temp.columns.duplicated()) and j == 0:
            print(f"Duplicate columns: {temp.loc[:, temp.columns.duplicated()].columns}")
        temp = temp.loc[:, ~temp.columns.duplicated()]  # removing duplicate columns
        # Before touch down
        temp = temp[temp["WEIGHT ON WHEELS"] == 1]
        # Correcting altitude using destination airport height
        temp["ALTITUDE"] = temp["PRESSURE ALTITUDE LSP"] - temp.loc[-500:, "PRESSURE ALTITUDE LSP"].min()
        # Calculating remaining distance to touch-down
        distance = integrate.cumtrapz(temp["GROUND SPEED LSP"] / 3600 * 1.15)  # Converting to mph and then mi/s + integral
        temp = temp.iloc[1:, :]  # Removing first row
        temp["DISTANCE_TO_LANDING"] = distance
        # temp["sKE"] = 0.5 * temp["TRUE AIRSPEED LSP"] ** 2  # specific kinetic energy (no mass)
        # temp["sPE"] = 32.17 * temp["ALTITUDE"]  # specific PE
        temp["flight_id"] = j
        temp["filename"] = flight

        temp["Anomaly"] = anomaly_tag

        if temp.isnull().sum().sum() > 0:
            print("Missing Data")
            missing_data_flight_id.append(j)
        DF_list.append(temp)
        if j % 500 == 0:
            with open(file_repeat_save, "wb") as f:
                pkl.dump(DF_list, f)
        if LIMIT is not None:
            if len(DF_list) == LIMIT:
                break

    print(f"\nTime taken: {time.time() - start}")
    print(f"Number of flights: {len(DF_list)}")
    return DF_list, missing_data_flight_id, no_file

def data_containers_inter_scale(dc1, dc2, data_set="validation", indx_filename=-3, **kw):
    # col_indices = []
    # for col in dc1.header:
    #     col_idx = np.where(col==dc2.header)[0]
    #     col_indices.append(col_idx)
    dc2.df = dc2.df[dc1.df.columns]
    dc2.header = dc1.header
    dc2.MIL_processing(**kw)
    dc2.scaler = dc1.scaler
    if data_set=="train":
        X_prime = dc2.trainX
        y_prime = dc2.trainY
    elif data_set=="validation":
        X_prime = dc2.valX
        y_prime = dc2.valY
    elif data_set=="test":
        X_prime = dc2.testX
        y_prime = dc2.testY

    if indx_filename is not None:
        X_prime = np.delete(X_prime, indx_filename, axis=2)

    X = dc2.normalizeData(use_loaded_df=False, data_set=data_set,
                                data=X_prime,
                               )
    return X.reshape(len(y_prime), -1, dc1.trainX.shape[-1]), y_prime

def interp_bad_data(data, columns, limit, greaterThan=False, **kw):
    for col in columns:
      if  not greaterThan:
        mask  = (data[col] < limit )
      else:
        mask  = (data[col] > limit )
      data[col][mask] = np.nan
      data[col] = data[col].interpolate(**kw, axis=0)
    return data