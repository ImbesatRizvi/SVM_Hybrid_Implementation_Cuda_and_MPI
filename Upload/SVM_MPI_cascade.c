#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define BILLION 1e9
#define FILE_READ_BUF_SIZE 100
#define DEFAULT_TOL 1e-3
#define DEFAULT_REGULARIZER_COEFF 100.0
#define DEFAULT_GAMMA 0.5
#define NUM_FEATURES 14
#define MAX_ITR 1e3

void print_mat(int **mat, int rows, int cols);
void print_arr(double *arr, int entries);
double kernel_eval(int *arr1, int *arr2, int arr_len, int offset, double gamma);

void svm_train(double **alpha, double *bias, int **train_cases, int num_train_cases, int num_features, double C,
		double gamma, int max_itr, double tol, int level, int rank);
void svm_level_init(double **alpha, int **train_cases, double **kernel_val_mat, 
			double *opt_cond, double gamma, int num_train_cases, int num_features, int level);
void bias_bound_index(double **alpha, double *opt_cond, int **train_cases, int num_train_cases, int num_features,
			double C, double *bias_low, double *bias_high, int *bias_low_indx, int *bias_high_indx);
void opt_cond_itr(int num_train_cases, double *opt_cond, double alpha_high, double alpha_high_prev,
			int high_label, int high_indx, double alpha_low, double alpha_low_prev, int low_label,
			int low_indx, double **kernel_val_mat);

void svm_test(double **alpha, double bias, int num_test_cases, int num_train_cases, int **train_cases, int test_start_indx,
		int **test_cases, double gamma, double **kernel_val_mat, int num_features, int *corr_pred_count);
	
int main(int argc,char **argv){

	int i, j, k;
	char file_read_buf[FILE_READ_BUF_SIZE]; // Buffer for file reading.
	int num_cases, num_train_cases, num_test_cases;
	
	int rank, n_procs;
	int tot_num_train_cases, tot_num_test_cases;

	int **train_cases, **test_cases;
	double **alpha, tol, C, gamma; // C is the Regularizer coeff., gamma is the multiplier in Gaussian Kernel exp{-gamma*||xi - xj||^2}.
	
	double bias;
	int corr_pred_count;
	int tot_corr_pred;

	//struct timespec train_start, train_stop, test_start, test_stop;
	double train_start, train_stop, test_start, test_stop;
	double train_duration, test_duration;
	
	int level = 0, max_itr;
	
	double **kernel_val_mat;

	char *file_name = "/home/secriz/PP_Project/Income_Classification_data.txt";
	FILE * file;

	if(argc > 1)
		file_name = argv[1];

	file = fopen(file_name, "r");
	fgets(file_read_buf, sizeof(file_read_buf), file);
	sscanf(file_read_buf,"%d %*[\n]", &num_cases);

	tot_num_train_cases = (argc > 2) ? atoi(argv[2]) * num_cases / 100 : num_cases / 2;
	tot_num_test_cases = num_cases - tot_num_train_cases;

	train_cases = (int **) malloc(tot_num_train_cases * sizeof(int*));
	for(i = 0; i < tot_num_train_cases; i++){
		train_cases[i] = (int *) malloc((NUM_FEATURES + 1) * sizeof(int)); // +1 for class label
		for(j = 0; j < (NUM_FEATURES + 1); j++)
			train_cases[i][j] = 0;
	}

	test_cases = (int **) malloc(tot_num_test_cases * sizeof(int*));
	for(i = 0; i < tot_num_test_cases; i++){
		test_cases[i] = (int *) malloc((NUM_FEATURES + 1) * sizeof(int)); // +1 for class label
		for(j = 0; j < (NUM_FEATURES + 1); j++)
			test_cases[i][j] = 0;
	}

	//printf("Total number of cases are %d.\n",num_cases);

	// Reading and storing train cases.
	for(i = 0; i < tot_num_train_cases; i++){
		fgets(file_read_buf, sizeof(file_read_buf), file);
		sscanf(file_read_buf, "%d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %*[\n]",
			&train_cases[i][0], &train_cases[i][1], &train_cases[i][2], &train_cases[i][3], &train_cases[i][4], &train_cases[i][5],
			&train_cases[i][6], &train_cases[i][7], &train_cases[i][8], &train_cases[i][9], &train_cases[i][10], &train_cases[i][11],
			&train_cases[i][12], &train_cases[i][13], &train_cases[i][14]);
	}

/*	printf("Train cases Matrix is:\n");
	print_mat(train_cases, num_train_cases, 15);
*/
	// Reading and storing test cases.
	for(i = 0; i < tot_num_test_cases; i++){
		fgets(file_read_buf, sizeof(file_read_buf), file);
		sscanf(file_read_buf, "%d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %d:%*d %*[\n]",
			&test_cases[i][0], &test_cases[i][1], &test_cases[i][2], &test_cases[i][3], &test_cases[i][4], &test_cases[i][5],
			&test_cases[i][6], &test_cases[i][7], &test_cases[i][8], &test_cases[i][9], &test_cases[i][10], &test_cases[i][11],
			&test_cases[i][12], &test_cases[i][13], &test_cases[i][14]);
	}
/*
	printf("Test cases Matrix is:\n");
	print_mat(test_cases, num_test_cases, 15);
*/

	fclose(file);
	
	tol = (argc > 3) ? atof(argv[3]) : DEFAULT_TOL;
	C = (argc > 4) ? atof(argv[4]) : DEFAULT_REGULARIZER_COEFF;
	gamma = (argc > 5) ? atof(argv[5]) : DEFAULT_GAMMA;
	max_itr = (argc > 6) ? atof(argv[6]) : MAX_ITR;
	
	MPI_Init(&argc, &argv);	
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Status status_s, status_r;
	MPI_Request request_s, request_r;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &n_procs);
	MPI_Barrier(comm);

	num_train_cases = (rank < (n_procs-1)) ? tot_num_train_cases/n_procs
				: tot_num_train_cases - (n_procs -1)*(tot_num_train_cases/n_procs);

	num_test_cases = (rank < (n_procs-1)) ? tot_num_test_cases/n_procs
				: tot_num_test_cases - (n_procs -1)*(tot_num_test_cases/n_procs);
				
	int train_start_indx = rank * tot_num_train_cases / n_procs;
	int test_start_indx = rank * tot_num_test_cases / n_procs;

	// Allocating space for storing Langrange Multiplier and initializing it to 0
	alpha = (double **) malloc(2 * sizeof(double *));
	alpha[0] = (double *) malloc((tot_num_train_cases+1) * sizeof(double));
	alpha[1] = (double *) malloc((tot_num_train_cases+1) * sizeof(double));
	alpha[0][0] = num_train_cases;
	alpha[1][0] = num_train_cases;
	
	for(i = 1; i <= num_train_cases; i++){
		alpha[0][i] = 0.0;
		alpha[1][i] = train_start_indx + i - 1;
	}
	
	int buff_size =  2*tot_num_train_cases + 2; //one extra for the count and the other extra for bias value for MPI_Bcast
	double *buff = (double *) malloc(buff_size * sizeof(double));
	
	int max_level = (int) log2(n_procs);
	int bcast_count;
	
	train_start = MPI_Wtime();
	while( level <= max_level ){
		
		if( rank % (1 << level) == 0 ){
			
			svm_train(alpha, &bias, train_cases, num_train_cases, NUM_FEATURES, C, gamma, /*kernel_val_mat,*/ max_itr, tol, level, rank);
			
			if( level != max_level ){
			
				if( rank % (1 << (level + 1)) == 0 ){
					
					k = 1;
					
					for(i = 1; i <= alpha[0][0]; i++){
						
						if(alpha[0][i] != 0.0){
							alpha[0][k] = alpha[0][i];
							alpha[1][k] = alpha[1][i];
							k++;
						}
					}
					alpha[0][0] = k-1;
					alpha[1][0] = k-1;

					MPI_Irecv(buff, buff_size, MPI_DOUBLE, rank + (1 << level), 0, comm, &request_r); //0 is the tag
					MPI_Wait(&request_r, &status_r);
					
					for(j = 1; j <= (int) buff[0]; j += 2){
					
						alpha[0][k] = buff[j];
						alpha[1][k] = buff[j+1];
						k++;
					}
					
					alpha[0][0] += (buff[0]/2);
					alpha[1][0] += alpha[0][0];
					
					num_train_cases = (int) alpha[0][0];
					
				}
			  	else {
				  	k = 1;
				  	for(i = 1; i <= (int) alpha[0][0]; i++){
				  		if(alpha[0][i] != 0.0){
				  			buff[k] = alpha[0][i];
				  			buff[k+1] = alpha[1][i];
				  			k += 2;
				  		}
				  	}
				  	buff[0] = 2*alpha[0][0];
				  	
				  	MPI_Isend(buff, (int) (buff[0] + 1), MPI_DOUBLE, rank - (1 << level), 0, comm, &request_s);
				  	MPI_Wait(&request_s, &status_s);
			  	}
		 	}
		}
		
		level++;	
	}
	
	//Prepare and send the learned alpha and bias to all the processors
	if(rank == 0){
  		k = 1;
  		for(i = 1; i <= (int) alpha[0][0]; i++){
  			if(alpha[0][i] != 0.0){
				buff[k] = alpha[0][i];
				buff[k+1] = alpha[1][i];
 				k += 2;
  			}
  		}
  		buff[0] = 2*alpha[0][0];
  		buff[k] = bias;
  		
  		bcast_count = (int) buff[0] + 2;
  	}
  	
  	MPI_Bcast(&bcast_count, 1, MPI_INT, 0, comm);
  	MPI_Barrier(comm);
  	MPI_Bcast(buff, bcast_count, MPI_DOUBLE, 0, comm);
  	
  	if(rank != 0){
  		k = 1;
		for(j = 1; j <= (int) buff[0]; j += 2){
			alpha[0][k] = buff[j];
			alpha[1][k] = buff[j+1];
			k++;
		}
		alpha[0][0] = (buff[0]/2);
		alpha[1][0] = alpha[0][0];
		bias = buff[ (int)buff[0] + 1 ];
		
		num_train_cases = (int) alpha[0][0];
	}
  	
	train_stop = MPI_Wtime();
	train_duration = train_stop - train_start;

	test_start = MPI_Wtime();
	
	svm_test(alpha, bias, num_test_cases, num_train_cases, train_cases, test_start_indx, test_cases, gamma, kernel_val_mat, NUM_FEATURES, &corr_pred_count);
	
	MPI_Reduce(&corr_pred_count, &tot_corr_pred, 1, MPI_INT, MPI_SUM, 0, comm);
	
	test_stop = MPI_Wtime();

	test_duration = test_stop - test_start;
	
	if(rank == 0){
		printf("Training time = %.3lf sec.\n", train_duration);
		printf("Testing time = %.3lf sec.\n", test_duration);
		printf("Correct Prediction = %d, Total Test Cases = %d, Percent Correct = %.2lf\n",
			tot_corr_pred, tot_num_test_cases, tot_corr_pred * 100.0 / tot_num_test_cases);
	}

	MPI_Finalize();
	return 0;
}

void svm_train(double **alpha, double *bias, int **train_cases, int num_train_cases, int num_features, double C,
		double gamma, int max_itr, double tol, int level, int rank){
	
	int i, j;
	double bias_low, bias_high, eta;
	int bias_low_indx, bias_high_indx;
	double alpha_high_prev, alpha_low_prev;
	
	double *opt_cond = (double *) malloc(num_train_cases * sizeof(double));

	double **kernel_val_mat = (double **) malloc(num_train_cases * num_train_cases * sizeof(double *));
	for(i = 0; i < num_train_cases; i++){
		kernel_val_mat[i] = (double *) malloc(num_train_cases * sizeof(double));
	}
	
	svm_level_init(alpha, train_cases, kernel_val_mat, opt_cond, gamma, num_train_cases, num_features, level);
	for(i = 0; i < max_itr; i++){
		bias_bound_index(alpha, opt_cond, train_cases, num_train_cases, 
			num_features, C, &bias_low, &bias_high, &bias_low_indx, &bias_high_indx);
		if( (bias_low - bias_high)/2 <= tol)
			break;

		eta = 2 * (1 - kernel_val_mat[bias_high_indx][bias_low_indx]);

		//to account for one count column in alpha
		bias_high_indx++;
		bias_low_indx++;

		alpha_low_prev = alpha[0][bias_low_indx];
		alpha_high_prev = alpha[0][bias_high_indx];

		alpha[0][bias_low_indx] += train_cases[(int)(alpha[1][bias_low_indx])][0] * (bias_high - bias_low) / eta;
		
		if(alpha[0][bias_low_indx] < 0.0)
			alpha[0][bias_low_indx] = 0.0;
		else if (alpha[0][bias_low_indx] > C)
			alpha[0][bias_low_indx] = C;

		alpha[0][bias_high_indx] += train_cases[(int)(alpha[1][bias_low_indx])][0]* 
				train_cases[(int)(alpha[1][bias_high_indx])][0] * (alpha_low_prev - alpha[0][bias_low_indx]);

		if(alpha[0][bias_high_indx] < 0.0)
			alpha[0][bias_high_indx] = 0.0;
		else if(alpha[0][bias_high_indx] > C)
			alpha[0][bias_high_indx] = C;

		opt_cond_itr(num_train_cases, opt_cond, alpha[0][bias_high_indx], alpha_high_prev,
			train_cases[(int)(alpha[1][bias_high_indx])][0], bias_high_indx-1, alpha[0][bias_low_indx], alpha_low_prev, 
			train_cases[(int)(alpha[1][bias_low_indx])][0], bias_low_indx-1, kernel_val_mat);
	}

	*bias = (bias_low + bias_high) / 2;
}

void svm_level_init(double **alpha, int **train_cases, double **kernel_val_mat, 
			double *opt_cond, double gamma, int num_train_cases, int num_features, int level){

	int i, j;

	for(i = 0; i < num_train_cases; i++){
		for(j = 0; j < num_train_cases; j++){
			if(i == j)
				kernel_val_mat[i][j] = 1.0;
			else if(i < j)
				kernel_val_mat[i][j] = kernel_eval(train_cases[(int)(alpha[1][i+1])], train_cases[j], num_features, 1, gamma);
			else
				kernel_val_mat[i][j] = kernel_val_mat[j][i];
			if(j == 0){
				opt_cond[i] = - train_cases[(int)(alpha[1][i+1])][0];
			}
			if(level != 0){
				opt_cond[i] += (alpha[0][i+1]*train_cases[(int)(alpha[1][i+1])][0]*kernel_val_mat[i][j]);
			}					
		}
	}
}

void bias_bound_index(double **alpha, double *opt_cond, int **train_cases, int num_train_cases, int num_features,
			double C, double *bias_low, double *bias_high, int *bias_low_indx, int *bias_high_indx){

	int j;
	int bias_low_first_set = 0;
	int bias_high_first_set = 0;
	for(j = 0; j < num_train_cases; j++){

		if( (alpha[0][j+1] > 0.0 && alpha[0][j+1] < C) || 
			(train_cases[(int)(alpha[1][j+1])][0] == 1 && alpha[0][j+1] == 0.0) || 
			(train_cases[(int)(alpha[1][j+1])][0] == -1 && alpha[0][j+1] == C) ){
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

		if( (alpha[0][j+1] > 0.0 && alpha[0][j+1] < C) ||
			(train_cases[(int)(alpha[1][j+1])][0] == 1 && alpha[0][j+1] == C) ||
			(train_cases[(int)(alpha[1][j+1])][0] == -1 && alpha[0][j+1] == 0.0)){

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

double kernel_eval(int *arr1, int *arr2, int arr_len, int offset, double gamma){
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

void opt_cond_itr(int num_train_cases, double *opt_cond, double alpha_high, double alpha_high_prev,
			int high_label, int high_indx, double alpha_low, double alpha_low_prev, int low_label,
			int low_indx, double **kernel_val_mat){

	int i;
	for(i = 0; i < num_train_cases; i++){
		opt_cond[i] += (alpha_high - alpha_high_prev) * high_label * kernel_val_mat[high_indx][i]
			+ (alpha_low - alpha_low_prev) * low_label * kernel_val_mat[low_indx][i];
	}
}

void svm_test(double **alpha, double bias, int num_test_cases, int num_train_cases, int **train_cases, int test_start_indx,
			int **test_cases, double gamma, double **kernel_val_mat, int num_features, int *corr_pred_count){

	int i, j = 1, num_SVs;
	double prediction;
	*corr_pred_count = 0;
	
	for(i = 0; i < num_test_cases; i++){
		prediction = bias;
		for(j = 0; j < num_train_cases; j++){
			if(alpha[0][j+1] != 0.0){
				prediction += ( alpha[0][j+1] * train_cases[(int)(alpha[1][j+1])][0] * kernel_eval(train_cases[(int)(alpha[1][j+1])], 
							test_cases[test_start_indx + i], num_features, 1, gamma) );
			}
		}

		prediction = (prediction > 0.0) ? 1 : -1;

		if( prediction == (double)test_cases[test_start_indx + i][0] ){
			*corr_pred_count += 1;
		}
	}
}
