#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define NUM_THREADS 512

__global__ void cuda_svm_level_init(double *alpha_gm, int *train_cases_gm, double *kernel_val_mat, 
			double *opt_cond, double gamma, int num_train_cases, int NUM_FEATURES, int level);

__device__ double kernel_eval(int *arr1, int *arr2, int arr_len, int offset, double gamma);

void bias_bound_index(double *alpha, double *opt_cond, int *train_cases, int num_train_cases, int NUM_FEATURES,
			double C, double *bias_low, double *bias_high, int *bias_low_indx, int *bias_high_indx);
			
__global__ void opt_cond_itr(int num_train_cases, double *opt_cond, double alpha_high, double alpha_high_prev,
			int high_label, int high_indx, double alpha_low, double alpha_low_prev, int low_label,
			int low_indx, double *kernel_val_mat);

__global__ void cuda_svm_test(double bias, int num_SVs, double *alpha_gm, int *SVs_gm, int *prediction, 
				int NUM_FEATURES, int *test_cases_gm, int num_test_cases, double gamma);
						
extern "C" void svm_train(double **alpha, double *bias, int **train_cases, int num_train_cases, int NUM_FEATURES, 
			double C, double gamma, double **kernel_val_mat, int max_itr, double tol, int level, int rank){

	int i, j;
	double bias_low, bias_high, eta;
	int bias_low_indx, bias_high_indx;
	double alpha_high_prev, alpha_low_prev;
	
	//double kernel_val;
	
	double *alpha_gm;
	cudaMalloc( (void **) &alpha_gm, num_train_cases * sizeof(double) );
	cudaMemcpy( alpha_gm, &alpha[0][1], num_train_cases*sizeof(double), cudaMemcpyHostToDevice );
	
	int *train_cases_gm;
	cudaMalloc( (void **) &train_cases_gm, num_train_cases * (NUM_FEATURES + 1) * sizeof(int) );
	
	
	int *level_train_cases = (int *) malloc(num_train_cases * (NUM_FEATURES + 1) * sizeof(int));
	for(i = 0; i < num_train_cases; i++){
		for(j = 0; j <= NUM_FEATURES; j++){
			level_train_cases[i*(NUM_FEATURES+1)+j] = train_cases[(int)(alpha[1][i+1])][j];
		}
	}
	cudaMemcpy( train_cases_gm, level_train_cases, num_train_cases * (NUM_FEATURES + 1) * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void **) kernel_val_mat, num_train_cases * num_train_cases * sizeof(double) );
	
	double *kernel_val_mat_cpu = (double *) malloc(num_train_cases * num_train_cases * sizeof(double));

	double *opt_cond;
	cudaMalloc( (void **) &opt_cond, num_train_cases * sizeof(double) );
	
	double *opt_cond_cpu = (double *) malloc(num_train_cases * sizeof(double));

	int num_blocks = (int) ceil((1.0*num_train_cases)/NUM_THREADS);
	
	dim3 dimGrid(num_blocks);
	dim3 dimBlock(NUM_THREADS);

	cuda_svm_level_init<<< dimGrid, dimBlock >>>(alpha_gm, train_cases_gm, *kernel_val_mat, 
						opt_cond, gamma, num_train_cases, NUM_FEATURES, level);

	cudaMemcpy( kernel_val_mat_cpu, *kernel_val_mat, num_train_cases*num_train_cases*sizeof(double), cudaMemcpyDeviceToHost );

	for(i = 0; i < max_itr; i++){
		cudaMemcpy( opt_cond_cpu, opt_cond, num_train_cases*sizeof(double), cudaMemcpyDeviceToHost );		
		
		bias_bound_index(&alpha[0][1], opt_cond_cpu, level_train_cases, num_train_cases, 
			NUM_FEATURES, C, &bias_low, &bias_high, &bias_low_indx, &bias_high_indx);
		if( (bias_low - bias_high)/2 <= tol)
			break;
			
		/*cudaMemcpy( &kernel_val, &kernel_val_mat[bias_high_indx*num_train_cases+bias_low_indx], 
				sizeof(double), cudaMemcpyDeviceToHost );
		eta = 2 * (1 - kernel_val);*/
		
		eta = 2 * (1 - kernel_val_mat_cpu[bias_high_indx * num_train_cases + bias_low_indx]);

		//to account for one count column in alpha
		bias_high_indx++;
		bias_low_indx++;

		alpha_low_prev = alpha[0][bias_low_indx];
		alpha_high_prev = alpha[0][bias_high_indx];

		alpha[0][bias_low_indx] += level_train_cases[(bias_low_indx-1)*(NUM_FEATURES+1)] * (bias_high - bias_low) / eta;
		
		if(alpha[0][bias_low_indx] < 0.0)
			alpha[0][bias_low_indx] = 0.0;
		else if (alpha[0][bias_low_indx] > C)
			alpha[0][bias_low_indx] = C;

		alpha[0][bias_high_indx] += level_train_cases[(bias_low_indx-1)*(NUM_FEATURES+1)] * 
				level_train_cases[(bias_high_indx-1)*(NUM_FEATURES+1)] * (alpha_low_prev - alpha[0][bias_low_indx]);

		if(alpha[0][bias_high_indx] < 0.0)
			alpha[0][bias_high_indx] = 0.0;
		else if(alpha[0][bias_high_indx] > C)
			alpha[0][bias_high_indx] = C;

		opt_cond_itr<<< dimBlock, dimBlock >>>(num_train_cases, opt_cond, alpha[0][bias_high_indx], alpha_high_prev,
			level_train_cases[(bias_high_indx-1)*(NUM_FEATURES+1)], bias_high_indx-1, alpha[0][bias_low_indx], alpha_low_prev, 
			level_train_cases[(bias_low_indx-1)*(NUM_FEATURES+1)], bias_low_indx-1, *kernel_val_mat);

	}
	//printf("Max Iter = %d\n",i);
	
	*bias = (bias_low + bias_high) / 2;
	
	cudaFree(alpha_gm);
	cudaFree(train_cases_gm);
	cudaFree(opt_cond);
	free(level_train_cases);
	free(kernel_val_mat_cpu);
	free(opt_cond_cpu);
}

extern "C" void svm_test(double **alpha, double bias, int num_test_cases, int num_train_cases, int **train_cases, int test_start_indx,
			int **test_cases, double gamma, int NUM_FEATURES, int *corr_pred_count){

	int i, j = 1, num_SVs;
	*corr_pred_count = 0;
	
	for(i = 1; i <= num_train_cases; i++){
		if(alpha[0][i] != 0.0){
			alpha[0][j] = alpha[0][i];
			alpha[1][j] = alpha[1][i];
			j++;
		}
	}
	num_SVs = j-1; //gives count of SVs.
	alpha[0][0] = num_SVs;
	alpha[1][0] = num_SVs;
	
	int *SVs = (int *) malloc(num_SVs * (NUM_FEATURES + 1) * sizeof(int));
	for(i = 0; i < num_SVs; i++){
		for(j = 0; j <= NUM_FEATURES; j++){
			SVs[i*(NUM_FEATURES+1)+j] = train_cases[(int)(alpha[1][i+1])][j];
		}
	}
	
	double *alpha_gm;
	cudaMalloc( (void **) &alpha_gm, num_SVs * sizeof(double) );
	cudaMemcpy( alpha_gm, &alpha[0][1], num_SVs *sizeof(double), cudaMemcpyHostToDevice );
	
	int *SVs_gm;
	cudaMalloc( (void **) &SVs_gm, num_SVs * (NUM_FEATURES + 1) * sizeof(int) );
	cudaMemcpy( SVs_gm, SVs, num_SVs * (NUM_FEATURES + 1) * sizeof(int), cudaMemcpyHostToDevice );
	
	int *local_test_cases = (int *) malloc(num_test_cases * (NUM_FEATURES + 1) * sizeof(int));
	for(i = 0; i < num_test_cases; i++){
		for(j = 0; j <= NUM_FEATURES; j++){
			local_test_cases[i*(NUM_FEATURES+1)+j] = test_cases[test_start_indx + i][j];
		}
	}
	
	int *test_cases_gm;
	cudaMalloc( (void **) &test_cases_gm, num_test_cases * (NUM_FEATURES + 1) * sizeof(int) );
	cudaMemcpy( test_cases_gm, local_test_cases, num_test_cases * (NUM_FEATURES + 1) * sizeof(int), cudaMemcpyHostToDevice );
	
	int *prediction;
	cudaMalloc( (void **) &prediction, num_test_cases * sizeof(int));
	
	int *prediction_cpu = (int *) malloc(num_test_cases * sizeof(int));
	
	int num_blocks = (int) ceil(1.0*num_test_cases/NUM_THREADS);
	
	dim3 dimGrid(num_blocks);
	dim3 dimBlock(NUM_THREADS);	
	
	cuda_svm_test<<< dimGrid, dimBlock >>>(bias, num_SVs, alpha_gm, SVs_gm,	prediction, 
				NUM_FEATURES, test_cases_gm, num_test_cases, gamma);

	cudaMemcpy( prediction_cpu, prediction, num_test_cases * sizeof(int), cudaMemcpyDeviceToHost );
	for(i = 0; i < num_test_cases; i++){
		if(prediction_cpu[i] == test_cases[i][0]){
			*corr_pred_count += 1;
		}
	}
	
	cudaFree(alpha_gm);
	cudaFree(SVs_gm);
	cudaFree(test_cases_gm);
	cudaFree(prediction);
	free(SVs);
	free(local_test_cases);
	free(prediction_cpu);
}

__global__ void cuda_svm_level_init(double *alpha_gm, int *train_cases_gm, double *kernel_val_mat, 
			double *opt_cond, double gamma, int num_train_cases, int NUM_FEATURES, int level){

	int i;
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	double kernel_val, loc_opt_cond;

	if(global_id < num_train_cases){
		for(i = 0; i < num_train_cases; i++){
	
			if(i == global_id){
				kernel_val = 1.0;
			}
			else{
				kernel_val = kernel_eval(&train_cases_gm[global_id*(NUM_FEATURES+1)], 
					&train_cases_gm[i*(NUM_FEATURES+1)], NUM_FEATURES, 1, gamma);
			}
			kernel_val_mat[global_id*num_train_cases+i] = kernel_val;
			if(i == 0){
				loc_opt_cond = - train_cases_gm[global_id*(NUM_FEATURES+1)];
			}
			if(level != 0){
				loc_opt_cond += alpha_gm[i]*train_cases_gm[i*(NUM_FEATURES+1)]*kernel_val;
			}
		}
	}
	opt_cond[global_id] = loc_opt_cond;
}

__device__ double kernel_eval(int *arr1, int *arr2, int arr_len, int offset, double gamma){
	double diff_norm_sqr = 0.0; // diff_norm_sqr is ||xi - xj||^2
	int m = offset;
	int n = offset;

	while(m <= arr_len && n <= arr_len){
		if(arr1[m] < arr2[n]){
			diff_norm_sqr += 1;
			m++;
		}
		else if(arr1[m] > arr2[n]){
			diff_norm_sqr += 1;
			n++;
		}
		else{
			m++;
			n++;
		}
	}
	while(m <= arr_len){
		diff_norm_sqr += 1;
		m++;
	}
	while(n <= arr_len){
		diff_norm_sqr +=1;
		n++;
	}
	return exp(-gamma * diff_norm_sqr);
}

void bias_bound_index(double *alpha, double *opt_cond, int *train_cases, int num_train_cases, int NUM_FEATURES,
			double C, double *bias_low, double *bias_high, int *bias_low_indx, int *bias_high_indx){

	int j;
	int bias_low_first_set = 0;
	int bias_high_first_set = 0;
	for(j = 0; j < num_train_cases; j++){

		if( (alpha[j] > 0.0 && alpha[j] < C) || 
			(train_cases[j*(NUM_FEATURES+1)] == 1 && alpha[j] == 0.0) || 
			(train_cases[j*(NUM_FEATURES+1)] == -1 && alpha[j] == C) ){
			if(bias_high_first_set == 0){
				bias_high_first_set = 1;
				*bias_high = opt_cond[j];
				*bias_high_indx = j;
			}
			else if(*bias_high > opt_cond[j]){
				*bias_high = opt_cond[j];
				*bias_high_indx = j;
			}
		}

		if( (alpha[j] > 0.0 && alpha[j] < C) ||
			(train_cases[j*(NUM_FEATURES+1)] == 1 && alpha[j] == C) ||
			(train_cases[j*(NUM_FEATURES+1)] == -1 && alpha[j] == 0.0)){

			if(bias_low_first_set == 0){
				bias_low_first_set = 1;
				*bias_low = opt_cond[j];
				*bias_low_indx = j;
			}
			else if(*bias_low < opt_cond[j]){
				*bias_low = opt_cond[j];
				*bias_low_indx = j;
			}
		}
	}
}

__global__ void opt_cond_itr(int num_train_cases, double *opt_cond, double alpha_high, double alpha_high_prev,
			int high_label, int high_indx, double alpha_low, double alpha_low_prev, int low_label,
			int low_indx, double *kernel_val_mat){

	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(global_id < num_train_cases){
		opt_cond[global_id] += (alpha_high - alpha_high_prev) * high_label * kernel_val_mat[high_indx*num_train_cases+global_id]
			+ (alpha_low - alpha_low_prev) * low_label * kernel_val_mat[low_indx*num_train_cases+global_id];
	}
}

__global__ void cuda_svm_test(double bias, int num_SVs, double *alpha_gm, int *SVs_gm, int *prediction, 
				int NUM_FEATURES, int *test_cases_gm, int num_test_cases, double gamma){

	int i;
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	double predict = bias;
	
	if(global_id < num_test_cases){
		for(i = 0; i < num_SVs; i++){
			predict += ( alpha_gm[i] * SVs_gm[i*(NUM_FEATURES+1)] * kernel_eval(&SVs_gm[i*(NUM_FEATURES+1)], 
				&test_cases_gm[global_id*(NUM_FEATURES+1)], NUM_FEATURES, 1, gamma) );
		}

		prediction[global_id] = (predict > 0.0) ? 1 : -1;
	}
}
