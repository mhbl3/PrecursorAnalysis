from .dataModel.dataProcessing import *
import argparse
from os.path import join
import sys

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()

parser.add_argument('--nominal-directory', type=str,
                    help='Relative directory of nominal flight data')

parser.add_argument('--adverse-directory', type=str,
                    help='Relative directory of adverse flight data')

parser.add_argument('--verbose', type=int, default=0,
                    help='Verbose to display the file names while processing data (set verbose=1)')

#TODO: Change var name
parser.add_argument('--filename', type=str, default="MyDataContainer",
                    help='Data container name')

parser.add_argument('--correlation-thres', type=float, default=0.0,
                    help='Dimensionality reduction of the data based on high correlation between feature')

parser.add_argument('--keep-features', type=str, default=None, nargs="+",
                    help='Feature to keep when performing reduction')

parser.add_argument('--del-features', type=str, default=None, nargs="+",
                    help='Feature to delete when performing reduction')

parser.add_argument('--target-feature', type=str, default=None,
                    help='Target feature name, will be changed to Anomaly')

parser.add_argument('--test-size', type=float, default=0.3,
                    help='Test set size')

parser.add_argument('--val-size', type=float, default=0.2,
                    help='Validation set size')

parser.add_argument('--anomaly-label', type=int, default=1, nargs="+",
                    help='Label for anomaly ')

parser.add_argument('--sampling-strategy', type=str, default=None, choices=["SMOTE",
                                                                            "random oversampler",
                                                                            "random undersampler",
                                                                            "SMOTETomek"],
                    help='Sampling strategy to work on imbalanced datasets')

parser.add_argument('--scaling-strategy', type=str, default="zscore", choices=["minmax",
                                                                            "zscore"],
                    help='Scaling strategy ')

args = parser.parse_args()


datacontainter = DataContainer(data=[args.nominal_directory,
                                     args.adverse_directory
                                     ],
                               verbose= int(args.verbose),
                               replace_name_target=args.target_feature,
                               fix_path=True
                                )
filename = join("Data", args.filename)

datacontainter.save(filename)

if args.correlation_thres > 0:
    datacontainter.correlated_feature_remover(correlation_threshold=args.correlation_thres,
                                        donotdelete=args.keep_features,
                                        force_dropped=args.del_features)
    print("List of features used:")
    print(datacontainter.header)
    datacontainter.save(filename)

datacontainter.MIL_processing(test_ratio=args.test_size,
                              val_ratio=args.val_size,
                              anomaly=args.anomaly_label,)
datacontainter.save(filename+"_MIL")

datacontainter.normalize_resample_data(sampling_method=args.sampling_strategy,
                                       scaling_method=args.scaling_strategy)
datacontainter.save(filename+"_MIL")
