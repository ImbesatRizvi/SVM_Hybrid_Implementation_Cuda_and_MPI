#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
void bias_bound_index(double *alpha, double *opt_cond, int **train_cases, 
	int num_train_cases, double C, double *bias_low, double *bias_high,
	int *bias_low_indx, int *bias_high_indx);

int main(int argc,char **argv){

	int i, j;
	char file_read_buf[FILE_READ_BUF_SIZE]; // Buffer for file reading.
	int num_cases, num_train_cases, num_test_cases;

	int **train_cases, **test_cases;
	double *alpha, *opt_cond, tol, C, gamma; // C is the Regularizer coeff., gamma is the multiplier in Gaussian Kernel exp{-gamma*||xi - xj||^2}.
	double bias_low, bias_high, eta;
	int bias_low_first_set, bias_high_first_set;
	int bias_low_indx, bias_high_indx, max_itr;
	double alpha_high_prev, alpha_low_prev;

	double **kernel_val_mat;

	double bias, prediction;
	int corr_pred_count;

	struct timespec train_start, train_stop, test_start, test_stop;
	double train_duration, test_duration;

	char *file_name = "/home/secriz/PP_Project/Income_Classification_data.txt";
	FILE * file;

	if(argc > 1)
		file_name = argv[1];

	file = fopen(file_name, "r");
	fgets(file_read_buf, sizeof(file_read_buf), file);
	sscanf(file_read_buf,"%d %*[\n]", &num_cases);

	num_train_cases = (argc > 2) ? atoi(argv[2]) * num_cases / 100 : num_cases / 2;
	num_test_cases = num_cases - num_train_cases;

	train_cases = (int **) malloc(num_train_cases * sizeof(int*));
	for(i = 0; i < num_train_cases; i++){
		train_cases[i] = (int *) malloc((NUM_FEATURES + 1) * sizeof(int)); // +1 for class label
		for(j = 0; j < (NUM_FEATURES + 1); j++)
			train_cases[i][j] = 0;
	}

	test_cases = (int **) malloc(num_test_cases * sizeof(int*));
	for(i = 0; i < num_test_cases; i++){
		test_cases[i] = (int *) malloc((NUM_FEATURES + 1) * sizeof(int)); // +1 for class label
		for(j = 0; j < (NUM_FEATURES + 1); j++)
			test_cases[i][j] = 0;
	}

	//printf("Total number of cases are %d.\n",num_cases);

	// Reading and storing train cases.
	for(i = 0; i < num_train_cases; i++){
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
	for(i = 0; i < num_test_cases; i++){
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

	// Allocating space for storing Langrange Multiplier and initializing it to 0
	alpha = (double *) malloc(num_train_cases * sizeof(double));
	for(i = 0; i < num_train_cases; i++)
		alpha[i] = 0.0;

//	print_arr(alpha, num_train_cases);

	// Allocating space for Optimality Condition Vector and initializing it to class labels from train set.
	opt_cond = (double *) malloc(num_train_cases * sizeof(double));
	for(i = 0; i < num_train_cases; i++)
		opt_cond[i] = - train_cases[i][0];

//	print_arr(opt_cond, num_train_cases);

	// Start training
	clock_gettime(CLOCK_REALTIME, &train_start);

	kernel_val_mat = (double **) malloc(num_train_cases * sizeof(double *));
	for(i = 0; i < num_train_cases; i++){
		kernel_val_mat[i] = (double *) malloc(num_train_cases * sizeof(double));
		for(j = 0; j < num_train_cases; j++){
			if(i == j)
				kernel_val_mat[i][j] = 1.0;
			else if(i < j)
				kernel_val_mat[i][j] = kernel_eval(train_cases[i], train_cases[j], NUM_FEATURES, 1, gamma);
			else
				kernel_val_mat[i][j] = kernel_val_mat[j][i];						
		}
	}

	int z,k;
	for(i = 0; i < max_itr; i++){
	
		bias_bound_index(alpha, opt_cond, train_cases, num_train_cases, C, &bias_low, &bias_high, &bias_low_indx, &bias_high_indx);
		if( (bias_low - bias_high)/2 <= tol)
			break;
			
		eta = 2 * (1 - kernel_val_mat[bias_high_indx][bias_low_indx]);

		alpha_low_prev = alpha[bias_low_indx];
		alpha_high_prev = alpha[bias_high_indx];

		alpha[bias_low_indx] += train_cases[bias_low_indx][0] * (bias_high - bias_low) / eta;
		
		if(alpha[bias_low_indx] < 0.0)
			alpha[bias_low_indx] = 0.0;
		else if (alpha[bias_low_indx] > C)
			alpha[bias_low_indx] = C;

		alpha[bias_high_indx] += train_cases[bias_low_indx][0] * train_cases[bias_high_indx][0] * (alpha_low_prev - alpha[bias_low_indx]);

		if(alpha[bias_high_indx] < 0.0)
			alpha[bias_high_indx] = 0.0;
		else if(alpha[bias_high_indx] > C)
			alpha[bias_high_indx] = C;

		for(j = 0; j < num_train_cases; j++){
			opt_cond[j] += ( (alpha[bias_high_indx] - alpha_high_prev) * train_cases[bias_high_indx][0]
						* kernel_val_mat[bias_high_indx][j]
					+ (alpha[bias_low_indx] - alpha_low_prev) * train_cases[bias_low_indx][0]
						* kernel_val_mat[bias_low_indx][j] ) ;
		}
	}
	clock_gettime(CLOCK_REALTIME, &train_stop);
	train_duration = ((train_stop.tv_sec - train_start.tv_sec) * BILLION + (train_stop.tv_nsec - train_start.tv_nsec)) / BILLION;

	printf("Max Iter = %d\n",i);

	bias = (bias_low + bias_high) / 2;

	corr_pred_count = 0;

	// Start Testing
	clock_gettime(CLOCK_REALTIME, &test_start);
	for(i = 0; i < num_test_cases; i++){
		prediction = bias;
		for(j = 0; j < num_train_cases; j++){
			if(alpha[j] != 0.0){
				prediction += ( alpha[j] * train_cases[j][0] * kernel_eval(train_cases[j], test_cases[i], NUM_FEATURES, 1, gamma) );
			}
		}

		prediction = (prediction > 0.0) ? 1 : -1;

		if( prediction == (double)test_cases[i][0] ){
			corr_pred_count += 1;
		}
	}
	clock_gettime(CLOCK_REALTIME, &test_stop);
	test_duration = ((test_stop.tv_sec - test_start.tv_sec) * BILLION + (test_stop.tv_nsec - test_start.tv_nsec)) / BILLION;

	printf("Training time = %.3lf sec.\n", train_duration);
	printf("Testing time = %.3lf sec.\n", test_duration);
	printf("Correct Prediction = %d, Total Test Cases = %d, Percent Correct = %.2lf\n",
		corr_pred_count, num_test_cases, corr_pred_count * 100.0 / num_test_cases);

	return 0;
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

void bias_bound_index(double *alpha, double *opt_cond, int **train_cases, int num_train_cases, 
			double C, double *bias_low, double *bias_high, int *bias_low_indx, int *bias_high_indx){

	int j;
	int bias_low_first_set = 0;
	int bias_high_first_set = 0;
	for(j = 0; j < num_train_cases; j++){

		if( (alpha[j] > 0.0 && alpha[j] < C) || 
			(train_cases[j][0] == 1 && alpha[j] == 0.0) || 
			(train_cases[j][0] == -1 && alpha[j] == C) ){
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
			(train_cases[j][0] == 1 && alpha[j] == C) ||
			(train_cases[j][0] == -1 && alpha[j] == 0.0)){

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

void print_mat(int **mat, int rows, int col){
	int i,j;
	for(i = 0; i < rows; i++){
		for(j = 0; j < col; j++)
			printf("%3d ", mat[i][j]);
		printf("\n");
	}
}

void print_arr(double *arr, int entries){
	int i;
	for(i = 0; i < entries; i++)
		printf("%.3lf ", arr[i]);
	printf("\n");
}
