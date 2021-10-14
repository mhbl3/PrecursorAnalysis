import os
import itertools as it
from ..tune.hyperparametertuning import tuning_wraper
from imblearn.under_sampling import RandomUnderSampler
import time
import multiprocessing as mp
from functools import partial
import argparse
import numpy as np
import torch
import pickle as pkl
import ast


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, nargs = "+",
                    help='learning rate')

parser.add_argument('--l2', type=float, nargs= "+", 
                    help='weight decay')

parser.add_argument('--ks',  type=int, nargs='+', action="append",
                    help='kernel size')
                    
parser.add_argument('--out-channels', type=int, nargs='+', action="append",
                    help='Channels')
                    
parser.add_argument('--mini-batch-percent', type=float, nargs = "+",
                    help='Percent for mini batch')
                    
parser.add_argument('--data-container', type=str,
                    help='data container path (after MIL)')

parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs')
                    
                    
parser.add_argument('--out-dir', type=str,
                    help='output directory')

parser.add_argument('--adopt-dir', type=str, nargs="+",
                    help='ADOPT directory')

parser.add_argument('--run-methods', default =[True, True, True], nargs='+',
                    help='Methods for hyper param')

parser.add_argument('--load-models', type=bool, default=False,
                    help='Load already trained trial models')

parser.add_argument('--use-stratisfy', type=bool, default=False,
                    help='True use stratisfied mini-batch strategy')

parser.add_argument('--aggregation', type=str, default="maxpool",
                    help='aggregation type for logistic regression model')

parser.add_argument('--dropout', type=float, nargs="+", default=0.0,
                    help='weight decay')

parser.add_argument('--gru-hidden', type=int, default=5,
                    help='Hidden layers for GRU')

parser.add_argument('--model-type', type=str, choices=["lr_binary","imdope_binary", "imdope_mc"],
                    help='Model type')

parser.add_argument('--use-cuda', type=bool, default=False,
                    help='True uses cuda')

parser.add_argument('--n-classes', type=int, default=2,
                    help='Number of classes')

parser.add_argument('--optimizer', type=str, choices=["SGD", "adam"], default="adam",
                    help='Optimizer to use')

args = parser.parse_args()

if args.load_models :
    load_models = True
else :
    load_models = False
    
if args.use_stratisfy:
    use_stratisfy = True
else :
    use_stratisfy = False

if args.use_cuda:
    device = "cuda:0"
else:
    device = "cpu"

run_methods = [ast.literal_eval(i) for i in args.run_methods]

###### Data Processing
#%%
if __name__ == "__main__":
    myseed = 45237552
    np.random.seed(120)
    torch.manual_seed(myseed)

    with open(args.data_container, "rb") as f:
        dm = pkl.load(f)

    # Shaping data for MIL
    # dm.MIL_processing(test_ratio=args.test_ratio, val_ratio=args.val_ratio, anomaly=[args.anomaly])

    # Normalizing data & Saving unormalized values for anomalies and flight ids
    # dc.normalize_resample_data(sampling_method=None)


    ####### Parameters definition
    #%%
    lr = args.lr #[.01]#, .01]
    # print(lr)               

    l2 = args.l2 #[0.01, 0.005] #[0.01, 0.005, 0.001]
    ks = args.ks #[[8, 5, 3]] #[[8, 5, 3],
                  # [10, 5 ,3],
                  # [8, 6, 4],
                  # [6, 3, 2]]
    # print(ks)
    out_channels = args.out_channels #[[16, 32, 64]] #[[16, 32, 64],
                     # [64, 128, 256],
                     # [16, 48, 144]]
    mini_batch = args.mini_batch_percent#[0.5]# [1, 0.5] #[1, 0.5, 0.25]
    last_layer = ["cnn"]#["dense", "cnn"]
    batch_norm = [True] #[True, False]
    epochs = [args.epochs]
    mini_b_strat = [use_stratisfy]

    aggregation = [args.aggregation]
    gru_hidden_size = [args.gru_hidden]
    dropout = [args.dropout]
    model_type = [args.model_type]
    n_classes = [args.n_classes]
    if model_type[0]=="imdope_binary":
        n_classes = [1]
    optimizer = [args.optimizer]



    if mini_b_strat[0] == False:
        params = {"learning_rate":lr,
                  "l2":l2,
                  "kernel_size":ks,
                  "n_filters":out_channels,
                  "batch_size_percent":mini_batch,
                  "n_classes":n_classes,
                  "use_batch_norm":batch_norm,
                  "optimizer":optimizer,
                  "epochs":epochs,
                  "gru_hidden_size":gru_hidden_size,
                  "dropout":dropout,
                  "model_type":model_type,
                  "aggregation":aggregation}
    else:
        params = {"learning_rate":lr,
                  "l2":l2,
                  "kernel_size":ks,
                  "n_filters":out_channels,
                  "batch_size_percent":mini_batch,
                  "n_classes":n_classes,
                  "use_batch_norm":batch_norm,
                  "optimizer":optimizer,
                  "epochs":epochs,
                  "use_stratified_batch_size":mini_b_strat,
                  "gru_hidden_size": gru_hidden_size,
                  "dropout": dropout,
                  "model_type": model_type,
                  "aggregation": aggregation
                  }


    n_processes = 2
    list_params = []
    procs = []
    load_trained_model = load_models
    combs = []
    dm.trainX, dm.valX = dm.trainX.astype(np.float64), dm.valX.astype(np.float64)
    data = {"X_train": dm.trainX.astype(np.float64), "y_train": dm.trainY,
            "data_model": dm,
            "X_test": dm.valX, "y_test": dm.valY}
#     print(dm.trainX.shape)
#     print(np.unique(dm.trainY))
    for name, _ in params.items():
        list_params.append(name)

    all_combinations = list(it.product(*(params[name] for name in list_params)))
    all_combinations = [list(tuple) for tuple in all_combinations]
    [i.insert(0, counter) for counter, i in enumerate(all_combinations)] # insert index of file
    
    print(f"Number of Combinations: {len(all_combinations)}")
    # %% Run trials
    if not os.path.exists("{}".format(args.out_dir)):
        os.makedirs("{}".format(args.out_dir))

    with open('{}/all_combinations.txt'.format(args.out_dir), 'a') as filehandle:
        filehandle.write(('%s\n' % str(list_params)))
        for listitem in all_combinations:
            filehandle.write('%s\n' % str(listitem))
    start = time.time()

    for run, combination in enumerate(all_combinations):
        path = os.path.join("{}".format(args.out_dir), f"Combination_{combination[0]}")
        if not load_trained_model:
            if os.path.exists(path):
                if os.path.exists(os.path.join(path,"completed.txt")):
                    continue
            else:
                os.makedirs(path)
        else:
            if not os.path.exists(os.path.join(path,"trial_model.pt")):
                continue

        myinput = {"combination": combination[1:],
                   "dm": dm, "list_params": list_params,
                   "path": path, "device": device}
        combs.append(myinput)
    # if ("cuda" in device) or ("cpu" in device):
    for iteration, comb in enumerate(combs):
        print(f"Starting combination_{iteration}")
        tuning_wraper(data, comb, load_trained_model, 
        run_methods=run_methods, path_to_ADOPT_run=args.adopt_dir)
    # else:
    #     pool = mp.Pool(processes=n_processes)#processes=n_processes)
    #     func = partial(tuning_wraper, data)
    #     pool.map(func, combs)
    #     pool.close()
    #     # pool.join()
    #     # proc = Process(target=fun, args=(myinput,))
    #     # procs.append(proc)
    #     # proc.start()

    #     # temp_dict = dict(zip(list_params, combination))
    #     #
    #     # mt = model_trial(path, dm, temp_dict)
    #     # mt.init_my_model(X_train.shape[2]-2, X_train.shape[1], device)
    #     # mt.fit(X_train=X_res[:-100], y_train=y_res[:-100], l2=temp_dict["l2"], learning_rate=temp_dict["learning_rate"],
    #     #        num_epochs=temp_dict["epochs"])
    #     # mt.evaluate_basic_metrics(X_test[:-100], y_test[:-100])
    #     # mt.evaluate_ADOPT_overlap()
    #     # mt.evaluate_best_feature_overlap()
    #     # mt.save_all()
    # # for proc in procs:
    # #     proc.join()
    end_time = time.time()-start
    with open(f'./{args.out_dir}/all_combinations.txt', 'a+') as filehandle:
        filehandle.write(f"Total Time (min): {end_time/60}\n")