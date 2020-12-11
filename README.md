# Work in Progress
This package is currently work in progress. The project is projected to be completed by May 2021. 

# PrecursorAnalysis
Identification and analysis of precursors of time-series using the Intelligent Methodology for the Discovery of Precursor of adverse Events (IM-DoPE). 
![IM-DoPE](img/img/IMDOPE.jpg)

# Training a model
The model can be train directly from the root of the repo using:
```
python -m imdope.train --model-type "mc" --lr 0.001 --l2 0.001 --ks 8 5 3 --out-channels 10 15 20 --use-stratisfy True --model-name "test_model.pt" --epochs 1 --data-container "./Data/new_dc_HS.pkl" --use-cuda True
```

