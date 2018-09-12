# Kickstarter_Analysis

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation <a name="motivation"></a>
I was interested in using Kickstarter project data from May 2009 to January 2018 to explore answers to the following questions:

1. Which categories have the best and worst success rates?
2. How does success rate relate to goal amount?
3. Does duration of a project influence success rate?
4. What features have the greatest influence on success?
5. Can we predict success of current/future projects?

## File Descriptions <a name="files"></a>
This repository includes two jupyter notebooks - one showing the exploratory analysis for the first three questions above (Kickstarter_analysis.ipynb), and another showing the machine learning steps to build a model for exploring questions 4 and 5 above (Kickstarter_ML.ipynb).

Additionally, there are two '.py' files: kickstarter_functions.py includes functions for running the exploratory analysis of the data, and kickstarter_ML.py includes functions necessary for building the model in the machine learning notebook.

## Results <a name="results"></a>
The main findings of the code can be found at the post available [here](https://medium.com/@cnspatino/this-is-how-to-increase-your-chances-of-having-a-successful-kickstarter-project-8ccd88eef489). Additional findings are discussed throughout the notebooks in the markdown cells.

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>
I'd like to thank Mickaël Mouillé for pulling the data from the Kickstarter platform and posting it on Kaggle. The data and its descriptions can be found on Kaggle [here](https://www.kaggle.com/kemical/kickstarter-projects).
