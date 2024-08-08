# Project Overview 

## bioSNN
This computational tool is part of the bachelor thesis "Exploration of Dynamics and Application of Spike-Timing-Dependent Plasticity". The purpose of bioSNN is to perform digit recognition on the MNIST dataset using biologically plausible learning mechanisms, mainly STDP, but also together with other mechanisms such as winner-takes-all, adaptive thresholding, stochasticity, and more (to be added). bioSNN should perform training, testing, and evaluating of various network configurations automatically and efficiently, while retaining high customizability. 

The results of the thesis demonstrate that the model, through STDP, can learn stable and differentiated representations of input data. The spike encoding mechanism plays a key role in this process. These findings together with the tool lay a foundation for further research into biologically plausible modeling of learning processes in spiking neural networks. 

## Installation 

To install the required packages run 

```zsh
pip install requirements.txt
```

## Usage 

The current implementation does not feature a GUI or CLI. To run an experiment change directories into the main folder 
```zsh
cd path/to/folder
```
and run
```zsh
python3 src/main.py
```
Parameters for each run can be modified in the respective dictionary. Running the file without changing parameters utilises the default parameters and optimal parameter setting specified in the thesis.

Results of the training will be displayed and subsequently saved to an experiment folder in the data folder. The naming scheme of the folders features key parameter settings from that run, model name, version, as well as a unique ID to allow for multiple experiments with the same parameter configuration to be saved. The experiment folder contains four subfolders and one .py file (as of version 1.03): 
- parameters: contains json files with all parameter settings
- plots: contains all plots generated during the run
- results: contains the output of the last evaluation
- weights: contains final weights and weight change file
- plot_weight_changes.py: script to generate interactive graphics showing the weight change during training, as heatmap and weight reconstruction

To view the weight change over the training time, execute the script plot_weight_changes.py in the experiment folder. A slider allows to scroll forward and backward through the training course. 

## Features 

The current version of bioSNN features the following mechanisms:

- adaptive threshold after Diehl & Cook (2015)
- hard/soft Winner-Takes-All
- "error" mechanism
- neuronal coding mechanism: Constant rate coding, linear Poisson coding, exponential Poisson coding

Potential future work could add:

- receptive fields
- recurrency
- delay
- neuronal backpropagation
- stochastic connections

NOT FUNCTIONAL:

Some features have been started but are not fully functional yet:
- specifying arbitrary network architectures
- changing neuron model

## License 

This software is licensed under the [MIT License](LICENSE).

## Citation 

For information on how to cite this software please refer to the [citation file](CITATION.cff).

## Contact 

For any inquiries please contact the author at:

nowaczyk2@uni-potsdam.de
