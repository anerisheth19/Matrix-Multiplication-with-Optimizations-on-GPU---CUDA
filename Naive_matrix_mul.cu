//ECGR 6090 Heterogeneous Computing Homework 1
// Problem 1  - Naive Matrix Multiplication on GPU
//Written by Aneri Sheth - 801085402


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define N 100
#define M 100
#define k 100


__global__ void matrix_mul(float *a, float *b, float *c){
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //where am I in matrix 
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	float temp = 0.0; //calculate sum 
	for(int i = 0;i < k;i++)
	{
		temp += a[row * k + i] * b[i * k + col]; //add and multiply
	}
	c[row * k + col] = temp; //final c matrix 
}

//Function to initialize matrices with random values
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; i++)
	for (int j = 0; j < size; j++)
		 *(data + i*size + j) = rand() % 10; 
}

//Function to display matrices 
void display_matrix(int size, float *matrix)
{
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			printf("Matrix = %f ",*(matrix + i*size + j));
		}
	}
}

int main(void)
{
	float *a, *b, *c; //CPU copies
	float *g_a, *g_b, *g_c; //GPU copies
	int matrix_size = N * M * sizeof(float);
	
	cudaEvent_t start, stop; //time start and stop
	float time;

	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
 
	//Allocate device memory
	cudaMalloc((void **)&g_a, matrix_size);
	cudaMalloc((void **)&g_b, matrix_size);
	cudaMalloc((void **)&g_c, matrix_size);

	//Allocate CPU memory
	a = (float *)malloc(matrix_size); randomInit(a, N);
	b = (float *)malloc(matrix_size); randomInit(b, M);
	c = (float *)malloc(matrix_size);

	//Copy CPU memory to GPU memory
	cudaMemcpy(g_a, a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, b, matrix_size, cudaMemcpyHostToDevice);

	//display_matrix(N,k,a);
	//display_matrix(k,M,b);

	//Set thread and grid dimensions 
	dim3 threadBlocks = dim3((int) std::ceil( (double) k/16 ),(int) std::ceil ( (double) k/16),1);
	//dim3 threadBlocks = dim3()
	dim3 threadsPerBlock = dim3(16,16,1);

	
	cudaEventRecord( start, 0 );
	//Call the kernel
	matrix_mul<<<threadBlocks,threadsPerBlock>>>(g_a,g_b,g_c);
	
	//display_matrix(N,M,g_c);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop);
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	//display_matrix(N,M,g_c);
	printf("GPU Execution Time = %f\n",time);

	//Copy from device to host
	cudaMemcpy(c, g_c, matrix_size, cudaMemcpyDeviceToHost);
	//display_matrix(N,M,c);
	//free cpu and gpu memory
	free(a); free(b); free(c);
	cudaFree(g_a); cudaFree(g_b); cudaFree(g_c);

	return 0;
}

