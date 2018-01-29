/*
*
*	Auth: Kapil Ashok Melwani
*	Email: alu0100883473@ull.edu.es
*	CUDA C Programming
*	Histograma Reduccion
*	Arquitecturas avanzadas y de Propósito Específico
*	29 - Jan - 2018
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
///////////////////////////////////////////////////////////////////////////
// defines
///////////////////////////////////////////////////////////////////////////
#define N 1000000
#define M 8

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// Kernel de operaciones dentro del histograma
///////////////////////////////////////////////////////////////////////////

__global__
void kernel(int *vector, int *local_histograma)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int posicion_histograma = 0;
    if(i < N){
    	posicion_histograma = vector[i]%M; //ValorElementoV mod M
    	atomicAdd(&(local_histograma[(blockIdx.x*M)+posicion_histograma]),1);
    	__syncthreads();
    }
}

///////////////////////////////////////////////////////////////////////////
// Kernel de inicialización del histograma
///////////////////////////////////////////////////////////////////////////

__global__
void histograma(int *local_histograma)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < N){
		if(threadIdx.x==0){
			for(int j=0;j<M;j++)
				local_histograma[blockIdx.x*M+j] = 0;
		}
	}
}

///////////////////////////////////////////////////////////////////////////
// Kernel de reduccion
///////////////////////////////////////////////////////////////////////////

__global__
void reduccion(int blockperGrid,int *local_histograma, int *reduced_histograma)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//requisito fundamental es que el nbloques del grid sea potencia de dos
	if(i < ((blockperGrid*M)/2)) //sumas una mitad y la otra
		atomicAdd(&(local_histograma[i]),local_histograma[(blockperGrid/2) * M + i]);
	if(i < M){ //copias el elemento en el histograma reducido
		reduced_histograma[i] = local_histograma[i];
		__syncthreads();
	}


}

///////////////////////////////////////////////////////////////////////////
// Función para mostrar histograma final
///////////////////////////////////////////////////////////////////////////

void mostrar_histograma(int *hst_vector,int suma)
{
	printf("\n\t\t\t\t\tHISTOGRAMA REDUCCION\n\n");
	printf("\n=====================================================================================================\n");
	for(int i=1;i<=M;i++)
		printf("|    %d   | ",i);
	printf("|   TOTAL  | ");
	printf("\n");
	for(int x=0;x<M;x++)
		printf("| %d | ",hst_vector[x]);
	printf("| %d  |",suma);

	printf("\n=====================================================================================================");

}

///////////////////////////////////////////////////////////////////////////
// rutina principal
///////////////////////////////////////////////////////////////////////////

int main(void)
{
	//vectores de números en host y device
	int *hst_vector;
	int *dev_vector;
	//vectores de histograma en host y device
	int *hst_reduced_histograma;
	int *dev_reduced_histograma;
	int *hst_local_histograma;
	int *dev_local_histograma;
	//errores Cuda
	cudaError_t error = cudaSuccess;
	//bloques e hilos
	int threadperBlock = 977; //para que block per grid sea 1024
	int blockperGrid = (N + threadperBlock-1)/threadperBlock;
	int blockperGrid_ = blockperGrid;
	//temporizadores
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsedTime = 0;
	//otros
	int sum_elements = 0;

	printf("Info: Reservando memoria para los vectores\n");
	//reservamos memoria en host vector
	hst_vector = (int*)malloc(N*sizeof(int));
	//comprobamos la reserva de memoria
	if(hst_vector == NULL){
		fprintf(stderr,"\nError en reserva de memoria de hst_vector, line ");
        exit(EXIT_FAILURE);
	}
	//reservamos memoria en host histograma reducido
	hst_reduced_histograma = (int*)malloc(M*sizeof(int));
	//comprobamos la reserva de memoria
	if(hst_reduced_histograma == NULL){
		fprintf(stderr,"\nError en reserva de memoria de hst_reduced_histograma");
        exit(EXIT_FAILURE);
	}
	//reservamos memoria en host histograma reducido
	hst_local_histograma = (int*)malloc(blockperGrid*M*sizeof(int));
	//comprobamos la reserva de memoria
	if(hst_local_histograma == NULL){
		fprintf(stderr,"\nError en reserva de memoria de hst_local_histograma");
        exit(EXIT_FAILURE);
	}
	//reservamos memoria en device de vector
	error = cudaMalloc((void**)&dev_vector,N*sizeof(int));
	//comprobamos la reserva de memoria
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en reserva de memoria de dev_vector. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	//reservamos memoria en device de histograma reducido
	error = cudaMalloc((void**)&dev_reduced_histograma,M*sizeof(int));
	//comprobamos la reserva de memoria
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en reserva de memoria de dev_reduced_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	//reservamos memoria en device de histograma local
	error = cudaMalloc((void**)&dev_local_histograma,blockperGrid*M*sizeof(int));
	//comprobamos la reserva de memoria
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en reserva de memoria de dev_local_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	//inicialización de valores del vector de numeros con valores entre 1 y 8
	srand ((int)time(NULL));
	for(int i=0; i<N; i++)
		hst_vector[i] = (int)(1+rand()%(256-1));

    // Copiamos los elementos del vector hst_vector en el vector dev_vector
	printf("Info: Copiando elementos de HOST -> DEVICE");
	error = cudaMemcpy(dev_vector,hst_vector,N*sizeof(int),cudaMemcpyHostToDevice);
	//comprobamos la copia
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en la copia de elementos de hst_vector a dev_vector. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaMemcpy(dev_local_histograma,hst_local_histograma,blockperGrid*M*sizeof(int),cudaMemcpyHostToDevice);
	//comprobamos la copia
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en la copia de elementos de hst_histograma a dev_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaMemcpy(dev_reduced_histograma,hst_reduced_histograma,M*sizeof(int),cudaMemcpyHostToDevice);
	//comprobamos la copia
	if(error != cudaSuccess){
		fprintf(stderr,"\nError en la copia de elementos de hst_reduced_histograma a dev_reduced_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	printf("Info: CUDA ejecutara %d hilos y %d bloques\n",threadperBlock,blockperGrid);

	//Ejecutaremos kernel y temporizadores
	printf("Info: Ejecutamos Kernel e iniciamos temporizadores\n");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//Iniciamos temporizador
	printf("Info: Temporizador iniciado\n");
	cudaEventRecord(start,0);
	histograma <<< blockperGrid,threadperBlock >>> (dev_local_histograma);
	kernel <<< blockperGrid,threadperBlock >>> (dev_vector,dev_local_histograma);
	for(int x = 0;x<log2((double)blockperGrid);x++){
		reduccion <<< blockperGrid,threadperBlock >>> (blockperGrid_,dev_local_histograma,dev_reduced_histograma);
		blockperGrid_ /= 2;
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("Error durante las llamadas a kernel. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	printf("Info: Duración de la creación del histograma completo %.2f sec\n",elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Info: Copiamos elementos de vectores dev -> host\n");
	error = cudaMemcpy(hst_vector,dev_vector,N*sizeof(int),cudaMemcpyDeviceToHost);
	//comprobamos la copia
	if(error != cudaSuccess){
		printf("\nError en la copia de elementos de dev_vector a hst_vector. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaMemcpy(hst_reduced_histograma,dev_reduced_histograma,M*sizeof(int),cudaMemcpyDeviceToHost);
	//comprobamos la copia
	if(error != cudaSuccess){
		printf("\nError en la copia de elementos de dev_reduced_histograma a dev_reduced_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	for(int x=0;x<M;x++)
		sum_elements += hst_reduced_histograma[x];
	if(sum_elements != N){
		printf("Error, en el histograma hay %d elementos de 5 millones\n",sum_elements);
	}else{
		printf("\n\nInfo: Ejecución finalizada\n");
		mostrar_histograma(hst_reduced_histograma,sum_elements);
	}

	error = cudaFree(dev_vector);
	if(error != cudaSuccess){
		printf("Error, fallo librando el vector dev_vector (error code %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaFree(dev_local_histograma);
	if(error != cudaSuccess){
		printf("Error, fallo librando el vector dev_local_histograma (error code %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaFree(dev_reduced_histograma);
	if(error != cudaSuccess){
		printf("Error, fallo librando el vector dev_reduced_histograma (error code %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	return 0;
}

