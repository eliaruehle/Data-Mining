# Possible steps for starting our work 

1. Clustering for all strategys, all three datasets and the __accuracy.csv__ dataset
   1. read in the csv-Data
   2. look for a concrete batch-size e.g. 3
   3. take all samples for one batch size which are of the same length
   4. put them into vectors, perform a PCA to lower the data dimension
   5. try out a k-means clustering for some k
   6. the k should be figured out via an experimental approach -> this can be done on a local machine since there is a managable amount of data 