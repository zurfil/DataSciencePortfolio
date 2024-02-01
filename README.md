# Data Science project portfolio.

This repository consists of a few projects showcasing understanding of the technologies,
methods and tools described in my CV.

Current projects are:
- Ensemble clustering with the presence of noise:\
  	My dissertation project.
	This project dives into the idea that small additions of noise into the dataset might force an increase in the diversity of the clustering ensemble, thus enabling it to capture more detail from the dataset resulting in an increase in accuracy. While final results remained not conclusive, there were several cases where this idea turned out to be correct.\
	Results can be found under EnsembleClustering/results path as .png graphs.

	Tools used: Python and relevant libraries.


- FIFA 22 player dataset exploratory and statistical analysis:\
	Use the FIFA 22 dataset to answer different questions about player characteristics-statistics correlation,
	for example:
	- How does the age affect the statistics?
    - Make a classifier to predict card quality (bronze, silver, gold) based on player statistics.
    - Make a regressor to predict player overall rating based on player statistics.
    - Make a classifier to predict player position (defender, midfielder, striker) based on player statistics.

	Tools used: Python and relevant libraries.


- Coffee orders and customer data:\
	From three different .csv files (customer data, sales data, product data) create a dashboard
	helpful to understand coffee sale trends. The current dashboard is not final and will be moved from Metabase to PowerBI soon.

	Tools used: PostgreSQL DB hosted on AWS RDB could service, 
	Metabase running as a local service in a Docker container.

  
- MNIST image recognition:\
	Popular dataset containing a set of handwritten numbers ranging from 0 to 9. The task is to make a deep learning algorithm
	which is able to recognise each number. 

	Tools used: Python (Tensorflow, please make sure to use Python 3.11)