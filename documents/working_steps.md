# TODO's

## General:
- there are 2 main questions that we can answer using the matrix-factorization clustering:
which strategies are similar (over all evaluated matrices on all datasets)
which strategies are similar on a specific dataset (interesting for Tony’s and Willi’s decision tree)
both approaches require a new form of underlying data
because all the tasks operate on large data we need to compute it as efficient as possible, therefore conduct the following libraries: pandarallel, multiprocessing, torch 

## Data-preprocessing for the matrix-factorization clustering: 
- in a single metric file: detect all NaN entries and use build in functions to interpolate values from the first entries which are not NaN (use 4-5 values for interpolation when available) -> every row should contain exactly 50 values, one for every cycle
- it’s necessary to figure out if an experiment (represented via the 4 hyperparameters that are already extracted in the current main function) was processed for all metric files on a dataset -> for task b) we need to delete all rows which belong to a hyperparameter configuration, which was not runned at all metrics, for task a) we have to proof over all datasets and metrics, whether an experiment was successfully run or not and clear the corresponding rows (it could beneficial to join single files based on the hyperparameters)
- all files have to be stacked together in one .csv file, it’s possible to stack them together by making the csv file large in horizontal direction or in vertical, we need to figure out what works best in case of efficient stacking
- make sure, that all data matrices for b) contain the data in the same order, otherwise the pattern recognition won’t work exactly (please document the structure)
- we need a function which cast the data into a big numpy ndarray matrix, ideally this is a 2-dimensional object, otherwise I have to use pytorch magic to lower the dimension and than the clustering isn’t that expressive; it’s also possible to cast it directly into an pytorch tensor of dimension, but therefore you need the numpy middle-step… also make sure, that you define the correct device (mps for macos, cuda for nvidia graphics, cpu for all other) and assign the tensor with tensor.to(device); additionally the entries should be of type torch.float32
- also note that for task a) we have one matrix for every strategy and for task b) we have a matrix for every strategy on a dataset; all matrices need to be collected in a container (also np.ndarray)
- integrate the new way of data processing into existing project structure (see the torch branch in new to get the code) -> maybe rewrite some things, the input.py file contains also some examples how to use pandarallel and multiprocessing 

## Analysis of AL-Performance:
- improve algorithm to measure how good an strategy performs for several metrics -> maybe figure out if there is a correlation between characteristics of a dataset and the performance for some metrics
- look at derivates of the samples and look how good the rise is for some experiments -> use the information from multiple samples to make a strong statement 
- Tony found a paper which describes which metrics are most meaningful to characterize the performance on several datasets, maybe his results help for the questions

## General Ideas and TODO’s:
- find a way to parallalize the current main, maybe use shared tensors, but thats advanced, i would recommend prior experience with tensorflow/pytorch for this
- set meaningful Logger Statements to track the progress on the HPC 
- (optional) think about what was the main contribution of the feature you implemented and how does it help to answer the research question -> mandatory for end presentation 
