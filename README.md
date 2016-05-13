# SVM_Hybrid_Implementation_Cuda_and_MPI
Parallel implementation of Support Vector Machines undertaken as course Project for the "Parallel Programming" course offered at Computational and Data Science (CDS) Department, IISc, Bangalore. Following three implementations have been done:

1) Sequential Implementation - QP in training phase solved using popular SMO (Sequential Minimal Optimization) technique.

2) Cascade SVM - Once through Cascaded SVM implementation using MPI is implemented with each SVM train using sequential SMO.

3) Hybrid SVM - Once through Cascaded SVM implementation using MPI is implemented with each SVM train using CUDA implemented SMO.

At present, cross validation phase has not been considered in the implementation. The kernel used is Gaussian Kernel while the dataset used is preprocessed Adult dataset with the task of classifying a person's income being greater or less than $50,000/year based on US census data.
