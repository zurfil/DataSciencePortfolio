Project: Improved ensemble clustering through the balance between noise-induced diversity and quality

-----------------------------------------------------------------------------------------------------

This directory consists of all the files necessary to run and reproduce the results of the project.
The project consists of 5 main files: 
	get_baselines.py, 
	functions_quality.py, 
	functions_diversity.py, 
	functions_balanced.py
	main.py.
All the source code for the project is stored there, only main.py is expected to be interacted with directly.
The interaction involves: 
	changing the set of datasets used for the experiment in the DATASETS viable,
	changing the set of floats representing a % noise level used for each dataset in the NOISE_LEVELS viable,
	running the main.py file to run the experiment.

The project contains few additional directories:
	data - directory there all .csv files for the project are stored
	data_graphs - directory where visualizations of all 2D datasets are stored with a .py file used to make them
	results - a directory that stores the results .csv files, visualizations of the results and a .py file used
          	    to make these visualizations,
	    	    testing - a subdirectory of the results directory where the results and visualizations of
				  the testing stage are stored.

Version numbers of all libraries used (for compatibility reasons) are stored in 'requirements.txt' file.