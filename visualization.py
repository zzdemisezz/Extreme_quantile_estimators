import torch
import numpy as np
import pandas as pd

from models import model_selection, get_best_crit, model_evaluation, model_evaluation_real, model_selection_real, model_selection_observations
from extreme.estimators import evt_estimators, evt_estimators_real
from extreme import visualization as simviz

import matplotlib.pyplot as plt


#Results for real dataset

#Model Selection algorithm for real data set
y = model_selection_real(distribution="be_firelosses", params={"evi":1., "rho":[-0.05]}, n_replications=1, metric="median")
print(y)

#Model evaluation algorithm for real dataset, prints the estimated quantiles
# 0.95 quantile
print(model_evaluation_real("2023-06-27_19-58-22", print_quantiles=True))
print(model_evaluation_real("2023-06-27_19-04-00", print_quantiles=True))
print(model_evaluation_real("2023-06-27_18-06-21", print_quantiles=True))
# 0.99 quantile
print(model_evaluation_real("2023-06-27_17-15-45", print_quantiles=True))
print(model_evaluation_real("2023-06-27_15-33-47", print_quantiles=True))
print(model_evaluation_real("2023-06-27_16-26-50", print_quantiles=True))
# 0.995 quantile
print(model_evaluation_real("2023-06-27_20-28-11", print_quantiles=True))
print(model_evaluation_real("2023-06-27_22-24-03", print_quantiles=True))
print(model_evaluation_real("2023-06-27_20-56-24", print_quantiles=True))

#Quantile estimates of the other extreme quantile estimators for the real dataset
print(evt_estimators_real(1, 1323, "be_firelosses", "{'evi': 1.0, 'rho': [-0.125]}", return_full=False,
                                  metric="median"))

#Mean, Median, RMSE and RMedSE plots of the real dataset
#You have to adjust the scale of the plot for different quantiles
simviz.xquantile_plot_real(NN="2023-06-27_18-06-21") # Quantile 0.95, J = 5
plt.show()
simviz.xquantile_plot_real(NN="2023-06-27_16-26-50") # Quantile 0.99, J = 5
plt.show()
simviz.xquantile_plot_real(NN="2023-06-27_20-56-24") # Quantile 0.995, J = 5
plt.show()

#Loglog plot of the real dataset
simviz.real_loglog_plot()
plt.show()

#Hill plot of the real dataset
simviz.real_hill_plot()
plt.show()

##################################################
##################################################

#Results for Simulated datasets

#Logspacing function plot
simviz.training_plot(k_anchor=100, show_as_video=False, epoch=471, saved=False, NN="2023-05-25_22-24-47-rep1")
plt.show()

#Model selection algorithm for the simulated datasets
y = model_selection(distribution="gpd", params={"evi":0.125, "rho":[-0.125]}, n_replications=10, metric="mean")
print(y)

#Model selection algorithm for simulated datasets with a different number of observation than 500
z = model_selection_observations(distribution="nhw", params={"evi":1., "rho":[-0.125]}, n_replications=10, n_data=1500, metric="mean")
print(z)

#Mean, Median, RMSE and RMedSE plots of the simulated datasets
simviz.xquantile_plot(NN="2023-07-01_19-42-27", metric="mean")
plt.show()






