# Active Regulatory Regions Predictions using Deep Learning and Bayesian Optimisation
This project has been developed for the Bioinformatics Course attended at the University of Milan, dealing with the prediction of tissue specific active regulatory regions of the human genome. The prediction task involves a number of steps to retrieve and preprocess the dataset, select relevant features, train and compare three Deep Learning models, namely a Feed-Forward Neural Network (FFNN), a Convolutional Neural Network (CNN) and a Multi-Modal Neural Network (MMNN).

The project is based on [this repository](https://github.com/LucaCappelletti94/bioinformatics_practice), maintained by Luca Cappelletti. Also, other Python packages are used to retreive the datasets, as indicated in `bioinformatics_report.pdf`.

# The Prediction Task
Gene expression and its connection with several diseases, including Alzheimer or cancer, can be understood through the prediction of active regulatory regions on specific tissues. In this project the prediction of the active regulatory components, namely enhancers and promoters, consists in a regression task and it involves the `GM12878` cell-line retrieved using the `HG38` assembly of the human genome.

# Structure of the project

Some figures are not included in `bioinformatics_report.pdf` due to limitations in the space, however all the information is in this repository:

- `barplots/`: contains the plots of the Mean Squared Error (MSE) for each model and for the enhancers and promoters prediction tasks;

- `feature_selection/`: contains the relevant features selected through the [BorutaSHAP](https://github.com/Ekeany/Boruta-Shap) algorithm;

- `images/`: contains the architectures for the models found through Bayesian Hyperparameter Optimisation (BHO), the tables of the training history for each model and the feature importance boxplots generated through BorutaSHAP;

- `model_checkpoints/`: contains the checkpoints of the training procedure for each model and for both prediction tasks;

- `optimizers_checkpoints/`: contains the hyperparamter configurations together with the associated metrics found by BHO;

- `results/`: contains the training history of the models for each prediction task, stored in .csv format. Also the Wilcoxon signed-rank test results are stored in the corresponding .csv file;