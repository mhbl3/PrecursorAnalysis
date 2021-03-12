import argparse
from .precursorModels.modelProcessing import precursor_model_BN, precursor_model_BN_mc, MILLR
import pickle as pkl
import numpy as np
from sklearn.metrics import classification_report
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model-type', type=str, choices=["lr_binary","imdope_binary", "imdope_mc"],
                    default ="imdope_binary",
                    help='Model type')

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')

parser.add_argument('--l2', type=float, default=0.001,
                    help='weight decay')

parser.add_argument('--dropout', type=float, nargs="+", default=0.0,
                    help='weight decay')

parser.add_argument('--ks', type=int, nargs='+', action="append",
                    default=[8,5,4],
                    help='kernel size')

parser.add_argument('--out-channels', type=int, nargs='+', action="append",
                    default= [10, 15, 20],
                    help='Channels')

parser.add_argument('--gru-hidden', type=int, default=5,
                    help='Hidden layers for GRU')

parser.add_argument('--fourth-layer', type=str, default="cnn", choices=["cnn", "dense"],
                    help='Fourth layer')

parser.add_argument('--mini-batch-percent', type=float, default=0.01,
                    help='Percent for mini batch')

parser.add_argument('--n_classes', type=int, default=2,
                    help='Number of classes')

parser.add_argument('--data-container', type=str,
                    help='data container path (after MIL)')

parser.add_argument('--optimizer', type=str, choices=["SGD", "adam"], default="adam",
                    help='Optimizer to use')

parser.add_argument('--include-val', type=bool, default=True,
                    help='Include validation set')

parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs')

parser.add_argument('--out-dir', type=str, default="./Data/dashlink/models",
                    help='output directory')

parser.add_argument('--aggregation', type=str, default="maxpool",
                    help='aggregation type for logistic regression model')

parser.add_argument('--use-stratisfy', type=str, default=False,
                    help='True use stratisfied mini-batch strategy')

parser.add_argument('--verbose', type=int, default=1,
                    help='Verbose setting')

parser.add_argument('--use-cuda', type=bool, default=False,
                    help='True uses cuda')

parser.add_argument('--load-model', type=bool, default=False,
                    help='Load pre-trained model')

parser.add_argument('--model-name', type=str, default="my_model.pt",
                    help='Model file name')

args = parser.parse_args()

ks = args.ks[-1]
out_channels = args.out_channels[-1]
gru_hidden_size = args.gru_hidden
fourth_layer = args.fourth_layer
aggregation = args.aggregation

batch_size_percent = args.mini_batch_percent
epochs = args.epochs
reg_l2 = args.l2
lr = args.lr
dropout = args.dropout

make_batch_stratified = args.use_stratisfy
device = "cuda:0" if args.use_cuda else "cpu"

verbose = args.verbose
model_name = args.model_name

with open(args.data_container, "rb") as f:
    dc = pkl.load(f)

X_train, y_train = dc.trainX, dc.trainY
if dc.using_val_data and args.include_val:
    X_val, y_val = dc.valX, dc.valY
else:
    X_val = y_val = None
X_test, y_test = dc.testX, dc.testY

# Define the container model
if not args.load_model:
    if args.model_type == "imdope_mc":
        model = precursor_model_BN_mc(input_size=X_train.shape[-1]-2,
                                  sequence_length=X_train.shape[1],
                                  kernel_size=ks, dropout=dropout, n_classes=args.n_classes,
                                  n_filters = out_channels, hidden_size=gru_hidden_size,
                                  fourth_layer=fourth_layer, device=device)

    elif args.model_type == "imdope_binary":
        model = precursor_model_BN(input_size=X_train.shape[-1] - 2,
                                      sequence_length=X_train.shape[1],
                                      kernel_size=ks, dropout=dropout,
                                      n_filters=out_channels, hidden_size=gru_hidden_size,
                                      fourth_layer=fourth_layer, device=device)

    elif args.model_type=="lr_binary":
        model = MILLR(input_dim=X_train.shape[-1] - 2,
                      flight_length=X_train.shape[1],
                      device=device,
                      aggregation=aggregation)


    # Train model
    if args.include_val:
        model.fit(batch_size=int(X_train.shape[0]*batch_size_percent),
                 X_train = X_train[:, :, :-2], y_train = y_train,
                 y_val=y_val, X_val=X_val.astype(np.float64)[:, :, :-2],
                 l2=reg_l2, learning_rate= lr, use_stratified_batch_size=make_batch_stratified,
                 clf = model, num_epochs = epochs, verbose= verbose,)
    else:
        model.fit(batch_size=int(X_train.shape[0] * batch_size_percent),
                  X_train=X_train[:, :, :-2], y_train=y_train,
                  l2=reg_l2, learning_rate=lr, use_stratified_batch_size=make_batch_stratified,
                  clf=model, num_epochs=epochs, verbose=verbose, )
else:
    model = torch.load(f"./Data/{model_name}",
                       map_location= device)
print("---"*50)
print("Training Results:")
if args.model_type == "imdope_binary":
    yhat = model.predict(X_train[:,:,:-2]).cpu().detach().numpy()
    yhat = (yhat > model.threshold).astype(int)
else:
    yhat = model.predict(X_train[:, :, :-2])[1].cpu().detach().numpy()
print(classification_report(y_train, yhat))

if args.model_type == "imdope_binary":
    yhat = model.predict(X_test[:,:,:-2]).cpu().detach().numpy()
    yhat = (yhat > model.threshold).astype(int)
else:
    yhat = model.predict(X_test[:, :, :-2])[1].cpu().detach().numpy()
if args.include_val:
    if args.model_type == "imdope_binary":
        yhat_val = model.predict(X_val[:, :, :-2]).cpu().detach().numpy()
        yhat_val = (yhat_val > model.threshold).astype(int)
    else:
        yhat_val = model.predict(X_val[:, :, :-2])[1].cpu().detach().numpy()
    print("---"*50)
    print("Validation Results:")
    print(classification_report(y_val, yhat_val))
print("---"*50)
print("Testing Results:")
print(classification_report(y_test, yhat))
if not args.load_model:
    model.save(f"./Data/{model_name}")