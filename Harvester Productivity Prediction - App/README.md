# Harvester Productivity Prediction App
The aim of this app is to predict the harvesting productivity of wood from planted forests (Eucalyptus), using data from mechanized harvesting (harvester) and meteorological data.


## How to use
- First, modify lines 224 and 289, replacing YOUR_PATH with the correct path on your computer, to access the files *.pkl.
- Linux users: Download this folder. Open the terminal and go to the Harvester Productivity Prediction - App folder.
  Then run the run.sh file
- Windows users: Download this folder. Run the python launcher and go to the Harvester Productivity Prediction - App folder.
  Then run: python -m streamlit run Harvester_Productivity_Prediction.py


## Python and Python modules requirements
- python>=3.8
- catboost==1.1.1
- pandas==1.5.3
- pickle==4.0
- streamlit==1.32.1

## System requirements
- Ubuntu 22.04
