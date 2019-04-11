//ECGR 6090 Heterogeneous Computing Homework 1
// Problem 2  - Naive Matrix Multiplication with Memory Coalescing on GPU
//Written by Aneri Sheth - 801085402


#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define N 100
#define M 100
#define K 100

__global__ void matrix_mul_coal(float *a, float *b, float *c)	{

	int row = blockIdx.y* blockDim.y+ threadIdx.y;		
	int col = blockIdx.x* blockDim.x+ threadIdx.x;		
	float temp = 0.0; //calculate sum
	for (int k = 0; k < K; k++)
		{
			temp += a[row * K + k] + b[k * K + col]; //add and multiply
		}
		
	c[row * K + col] = temp; //final c matrix
}

//Function to initialize matrices with random values
void randomInit (float *data, int size)	
{
	for (int i = 0; i <  size; i++) 
		for (int j = 0; j < size; j++) 
			*(data + i * size + j) = rand() % 1024; 
}

//Function to display matrices 
void display_matrix (float *matrix, int size) 
{
	for (int i = 0; i <  size; i++) 
		for (int j = 0; j < size; j++) 
			printf("%d ", *(matrix + i * size + j));
}

int main()	
{
	
	float *a, *b, *c, *bt; //CPU copies
	float *g_a, *g_b, *g_c;  //GPU copies 
	int matrix_size = N * M * sizeof(float);
	
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocate CPU memory
	a = (float *) malloc(matrix_size);	randomInit(a, N);
	b = (float *) malloc(matrix_size);	randomInit(b, M);
	bt = (float *) malloc(matrix_size);
	c = (float *) malloc(matrix_size);

	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			*(bt + i * M + j) = *(b + j * M + i);

	display_matrix (a, N);
	display_matrix (b, M);
	display_matrix (bt, M);

	//Allocate GPU memory 
	cudaMalloc((void **) &g_a, matrix_size);
	cudaMalloc((void **) &g_b, matrix_size);
	cudaMalloc((void **) &g_c, matrix_size);

	//Copy from CPU memory to GPU memory
	cudaMemcpy( g_a, a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy( g_ b, bt, matrix_size, cudaMemcpyHostToDevice);

	//Set thread and grid dimensions
	dim3 tBlock(16, 16);
	dim3 Grid((N + 16 - 1)/dimBlock.x, (M + 16 - 1)/dimBlock.y);

	cudaEventRecord( start, 0 );

	//Call kernels
	matrix_mul_coal<<< Grid, tBlock >>> (g_a, g_b, g_c);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("GPU Execution Time = %f\n",time);
	
	//Copy from device to host
	cudaMemcpy( c, g_c, matrix_size, cudaMemcpyDeviceToHost);

	//free cpu and gpu memory
	free(a); free(b); free(c);
	cudaFree(g_a); cudaFree(g_b); cudaFree(g_c);

	return 0;

}
