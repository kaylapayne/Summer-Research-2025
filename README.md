# Summer Research 2025: Decaying Dark Matter Modeling

---

## Overview

This repository contains the code and analysis developed during my summer research project focused on modeling decaying dark matter (DM) and its effects on cosmological observables. The work includes:

- Coupled Ordinary Differential Equations (ODEs) for DM decay modeling
- Anisotropic and isotropic analyses of the 2-point correlation function (2PCF)
- Data processing and visualization for DESI datasets
- Hubble parameter comparisons between measured data and models

## Prerequisites

Ensure you have Python 3.x installed along with the necessary libraries. It's recommended to use a virtual environment:

    python -m venv desidr1_env
    source desidr1_env/bin/activate  # On Windows use `desidr1_env\Scripts\activate`

## Installation 

Clone the repository:

    git clone https://github.com/kaylapayne/Summer-Research-2025.git
    cd Summer-Research-2025

Install required packages:

    pip install -r requirements.txt

---

## Data

Processed data files are located in the data/ directory. These files are used as inputs for the analysis scripts. Raw DESI data and random files are not tracked by Git to avoid large file storage.

---

## Decaying Dark Matter Models

#### HzPlotBasic.py
This script shows the comparison between the standard LCDM model and a decaying DM model in terms of the Hubble parameter

#### DDM_LCDM_DiffLifetimes.py
This script compares a decaying dark matter model to the standard LCDM model, while also showing multiple decay lifetimes and the fractional difference of each model in comparison to the LCDM model

#### H_InteractiveGammaAndFraction.py
This script creates a link to an interactive visual representation of the comparison between the standard LCDM model and the decaying DM mode, the user can change the fraction of DM that decays along with the decay rate constant (Gamma)

---

## Data Analysis

#### DESI_2PCF_Analysis
This script reads in raw DESI data files and you can choose whether to do an isotropic or anisotropic analysis of the 2-point correlation between points in the data set. The analysis requires data sets of random points in the same shape as that of the data to ensure correct analysis the BAO data.

---

## Model-Data Comparison

#### Data_DDM_Comparison.py
This script essentially does everything, it reads in the analysed data saved in files by the DESI_2PCF_Analysis.py script and plots both the 2-point correlation function to see the BAO peaks, it fits a gaussian to those peaks, and it then plots decaying DM models alongside the standard LCDM and processed data points as a sound horizon scaled Hubble parameter comparison over recent redshift history. Finally, this script also shows the fractional difference of all models and data in comparison to the standard LCDM expansion model. 

---

## Fish Plots

This folder contains scripts showing the standard LCDM model, a decaying DM model, and an interactive decaying DM model in terms of energy density of each species over redshift history

---

## Figures

This folder contains some fun figures and animations to show the datasets along with a script to generate a figure showing the energy densities as percentages today