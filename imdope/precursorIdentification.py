from .precursorModels.modelProcessing import *
from .dataModel.dataProcessing import *
from imdope.utils import plot_feature_effects_no_effects
import argparse
from os.path import join
import torch
import pickle as pkl

parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, default="test_model.pt",
                    help='Model file name')

parser.add_argument('--filename', type=str, default="MyDataContainer_MIL.pkl",
                    help='Data container name')

parser.add_argument('--flight-id', type=int,
                    help='Flight id')

parser.add_argument('--anomaly', type=int, default=1,
                    help='Anomaly')

parser.add_argument('--use-cuda', type=bool, default=False,
                    help='Use cuda')


args = parser.parse_args()
model_name = args.model_name
device = "cuda:0" if args.use_cuda else "cpu"
dc_path = args.filename
flight_id = args.flight_id
anomaly = args.anomaly
out_folder = f"./Data/Flight_{flight_id}_Anomaly{anomaly}"

with open(dc_path, "rb") as f:
    datacontainer = pkl.load(f)

flight = datacontainer.df[(datacontainer.df.flight_id==flight_id ) & (datacontainer.df.Anomaly==anomaly)]

X = np.concatenate((datacontainer.trainX[datacontainer.trainY.flatten()==anomaly, : , :],
                    datacontainer.valX[datacontainer.valY.flatten()==anomaly, :, :],
                    datacontainer.testX[datacontainer.testY.flatten()==anomaly, :, :]
                    ), axis=0)
idx_flight = np.where(X[:, 0, -2].flatten()==flight_id)[0][0]

x = X[idx_flight, :, :]

model = torch.load(f"./Data/{model_name}",
                       map_location= device)

# Doubling so that the extract of precursor does not results in an exception
x = np.tile(x, (2,1,1))
out = model.predict(x[:, :, :-2])[0]
print(f"Probability of event: {out.cpu().detach().numpy()}")
if out > model.threshold:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model.show_feature_importance(datacontainer.header,
                                  flight_id=0,
                                  class_interest=anomaly,
                                  plot_save=os.path.join(out_folder,"predictors_ranking.png"))

    model.plot_feature_effects(datacontainer.max_len, datacontainer.header,
                            flight_id=0,
                            save_path= out_folder,
                            class_interest = 0 if model.n_classes == 1 else anomaly,
                            show_precursor_range=True,
                            ticks_on = False
                            )
    datacontainer.nom_flights_mean_std()
    plot_feature_effects_no_effects(model.proba_time, 0, flight.values,
                                    datacontainer.header, datacontainer.mus, datacontainer.sigmas,
                                    c=3, ytick_one=True, show_precursor_range=True,
                                    save_path=os.path.join(out_folder,"flight_parameters.pdf"),
                                    ticks_on=False)