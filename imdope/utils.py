import matplotlib.pyplot as plt
import numpy as np
from .dataModel.dataProcessing import *
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
import warnings 
import torch
from sklearn.decomposition import PCA


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
                         save_path=None):
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
                  limiting_feature= "ALTITUDE",
                  limiting_value = 1000):
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
            float_cols = df.select_dtypes(include=["float32",
                                                   "float64"]).columns # Assumes numerical data is of type

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
                        temp[col] = df[col].iloc[0]
                        warnings.warn("This function does not work on categorical variables")
            
            # inversing distance to make it the distance left and setting a zero
            temp["DISTANCE_TO_LANDING"] = abs(
                temp.DISTANCE_TO_LANDING - temp.DISTANCE_TO_LANDING.iloc[-1])  # distance remaining
            # Filtering
            temp = temp[temp.DISTANCE_TO_LANDING <= distance_away]  # last 20 mi

            list_df[i] = temp
        except KeyboardInterrupt:
            raise
        except  :
            # Handling flights that had an issue
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
    temp_nom = df[(df.Anomaly == 0)]
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