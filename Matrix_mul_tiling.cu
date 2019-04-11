//ECGR 6090 Heterogeneous Computing Homework 1
// Problem 3  - Matrix Multiplication with Tiling on GPU
//Written by Aneri Sheth - 801085402
//Reference taken from:  https://github.com/yogesh-desai/TiledMatrixMultiplicationInCUDA 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define N 100
#define M 100
#define K 100
#define tile_size 16


__global__ void matrix_mul_shared(float *a, float *b, float *c) {
	
	__shared__ int a_tile[tile_size][tile_size]; 		//define shared memory tile for matrix a
	__shared__ int b_tile[tile_size][tile_size];		//define shared memory tile for matrix b

    int row = blockIdx.y * tile_size + threadIdx.y;	//where am I 
	int col = blockIdx.x * tile_size + threadIdx.x;	

	float temp = 0.0; //store sum
    int tileIdx; 

	//Load one tile of A and one tile of B into shared memory
	for (int s = 0; s < gridDim.x; s++) {
		tileIdx = row * K + s * tile_size + threadIdx.x;

		if(tileIdx >= K*K)
			a_tile[threadIdx.y][threadIdx.x] = 0;	//check if K is divisible by tile size 
		else
			a_tile[threadIdx.y][threadIdx.x] = a[tileIdx];
	

		tileIdx = (s * tile_size + threadIdx.y) * K + col;

		if(tileIdx >= K*K)
			b_tile[threadIdx.y][threadIdx.x] = 0; 	//check if K is divisible by tile size 
		else
			b_tile[threadIdx.y][threadIdx.x] = b[tileIdx];
			
		__syncthreads(); //to ensure all data is copied into threads 

		for (int j = 0; j < tile_size; j++)
			temp += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x]; //add and multiply

		__syncthreads(); //to ensure computation is stored in threads 
		
	}
	
	if(row < K && col < K) 	
		c[row * K + col] = temp; //store the result in output matrix c
    	
}

//Function to initialize matrices with random values
void randomInit (float *data, int size)	
{
	for (int i = 0; i <  size; i++) 
		for (int j = 0; j < size; j++) 
			*(data + i * size + j) = rand() % 1024; 
}


int main()	{
	
	
	float *a, *b, *c; //CPU copies
	float *g_a, *g_b, *g_c;  //GPU copies 
	int matrix_size = N * M * sizeof(float);
	
	cudaEvent_t start, stop;
	float time;

	//Start the cuda timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocate CPU memory
	a = (float *) malloc(matrix_size);	randomInit(a, N);
	b = (float *) malloc(matrix_size);	randomInit(b, M);
	c = (float *) malloc(matrix_size);

	//Allocate GPU memory 
	cudaMalloc((void **) &g_a, matrix_size);
	cudaMalloc((void **) &g_b, matrix_size);
	cudaMalloc((void **) &g_c, matrix_size);

	//Copy from CPU memory to GPU memory
	cudaMemcpy( g_a, a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy( g_ b, b, matrix_size, cudaMemcpyHostToDevice);

	//Set thread and grid dimensions
	dim3 tBlock(16, 16);
	dim3 Grid((N + 16 - 1)/dimBlock.x, (M + 16 - 1)/dimBlock.y);

	cudaEventRecord( start, 0 );

	//Call kernels
	matrix_mul_shared<<< Grid, tBlock >>> (g_a,g_b,g_c);

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
