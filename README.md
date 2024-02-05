# Studies on Electronegative Impurities

This project delves into detailed studies concerning electronegative impurities, focusing on understanding, analyzing, and perhaps mitigating their effects. The analysis is conducted using Python, with a series of dedicated functions and comprehensive data exploration encapsulated within a Jupyter notebook.

## Repository Contents

- `Electronegative_impurities_studies.ipynb`: This Jupyter notebook contains the Python code along with detailed commentary and analysis. The notebook is structured to facilitate a clear understanding of the procedures, results, and conclusions of the study.

- `elec_neg_functions.py`: This Python script contains various functions that are used within the notebook. These functions perform calculations, data processing, and analysis related to electronegative impurities.

- `libray.json`: This file contains all the various data values that are used within the python functions and the notebook. These values are coming from multiple published papers for the most part of them.

## Data Structure in `library.json`

The `library.json` file includes comprehensive data sets used for analysis, structured into three main categories: `Material`, `System`, and `Gas`. Each category contains specific information relevant to the study of electronegative impurities.

### Material
This category includes various materials with their respective properties for different solutes. Properties include:
- `Diffusion Constant`: Given in cm²/s.
- `Solubility`: Given as a fraction.
- `Activation Energy`: Given in eV.

The material properties are primarily sourced from the following research papers:
- [Teflon, Teflon Yale, Viton, and PE1 Properties](http://arxiv.org/abs/1703.09144)

### System
Describes different systems with attributes such as mass, volume, surface area, and thickness. Each system entry contains:
- `Xenon Mass`: Given in grams.
- `Field Factor`: Specific to the system, given in seconds (refer to "Screening for Electronegative Impurities" for more details).
- Material attributes: `Volume` (liters), `Surface Area` (cm²), `Thickness` (cm).

### Gas
Provides details about various gases, including:
- `Abundance in Air`: Fractional representation.
- `Molar Mass`: Given in g/mol.

## Getting Started

To get started with the project, you need to have Jupyter Notebook installed, preferably through the Anaconda distribution which provides Jupyter and other useful scientific Python libraries pre-installed.

1. Clone this repository to your local machine or download it as a zip file.
2. Navigate to the repository folder.
3. Open `Electronegative_impurities_studies.ipynb` in Jupyter Notebook.
4. The notebook is executable, and cells can be run individually. The analysis and comments within the notebook provide information on what each cell does.

## Prerequisites

- Python 3.12 (older versions may not work properly)
- Jupyter Notebook
- Necessary Python libraries (can be downloaded using `pip install -r requirements.txt`)

## How to Use

The notebook is self-contained and provides detailed explanations for each step of the analysis. To run the entire notebook:

1. Open the Jupyter Notebook application. It should automatically open in your browser.
2. In the application, navigate to the project directory and open `Electronegative_impurities_studies.ipynb`.
3. You can run the notebook step by step by clicking on each cell and pressing `Shift + Enter`, or you can run the entire notebook by clicking `Kernel` > `Restart & Run All` in the menu.

Please note that some cells may depend on the outputs of earlier cells, so it is recommended to run them in order.

## Functions in `elec_neg_functions.py`

This file contains multiple utility functions used in the analysis. Each function is documented with comments explaining what it does, its parameters, and its return value. These functions are imported and used within `Electronegative_impurities_studies.ipynb` for various calculations and data manipulations.

## Authors

- Antoine Amy

## Acknowledgments

- Ako Jamil, David Moore, Sierra Wilde & the nEXO collaboration
