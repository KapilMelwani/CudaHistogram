/*
*
*	Auth: Kapil Ashok Melwani
*	Email: alu0100883473@ull.edu.es
*	CUDA C Programming
*	Histograma
*	Arquitecturas avanzadas y de Propósito Específico
*	29 - Jan - 2018
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
///////////////////////////////////////////////////////////////////////////
// defines
///////////////////////////////////////////////////////////////////////////
#define N 5000000
#define M 8

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// Kernel de operaciones dentro del histograma
///////////////////////////////////////////////////////////////////////////

__global__
void kernel(int *vector, int *histograma)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int posicion_histograma = 0;
    if(i < N){
    	posicion_histograma = vector[i]%M; //ValorElementoV mod M
    	atomicAdd(&(histograma[posicion_histograma]),1);
    }
}

///////////////////////////////////////////////////////////////////////////
// Kernel de inicialización del histograma
///////////////////////////////////////////////////////////////////////////

__global__
void histograma(int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < N){
		if(i==0){
			for(int j=0;j<M;j++)
				histo[j] = 0;
		}
	}
}

///////////////////////////////////////////////////////////////////////////
// Función para mostrar histograma final
///////////////////////////////////////////////////////////////////////////

void mostrar_histograma(int *hst_vector,int suma)
{
	printf("\n\t\t\t\t\tHISTOGRAMA\n\n");
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
	int *hst_histograma;
	int *dev_histograma;
	//errores Cuda
	cudaError_t error = cudaSuccess;
	//bloques e hilos
	int threadperBlock;
	int blockperGrid;
	//temporizadores
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsedTime;
	//comprobación final
	int sum_elements = 0;

	printf("Info: Reservando memoria para los vectores\n");
	//reservamos memoria en host vector
	hst_vector = (int*)malloc(N*sizeof(int));
	//comprobamos la reserva de memoria
	if(hst_vector == NULL){
		printf("\nError en reserva de memoria de hst_vector");
        exit(EXIT_FAILURE);
	}
	//reservamos memoria en host histograma
	hst_histograma = (int*)malloc(M*sizeof(int));
	//comprobamos la reserva de memoria
	if(hst_histograma == NULL){
		printf("\nError en reserva de memoria de hst_histograma");
        exit(EXIT_FAILURE);
	}
	//reservamos memoria en device de vector
	error = cudaMalloc((void**)&dev_vector,N*sizeof(int));
	//comprobamos la reserva de memoria
	if(error != cudaSuccess){
		printf("\nError en reserva de memoria de dev_vector. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	//reservamos memoria en device de histograma
	error = cudaMalloc((void**)&dev_histograma,M*sizeof(int));
	//comprobamos la reserva de memoria
	if(error != cudaSuccess){
		printf("\nError en reserva de memoria de dev_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
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
		printf("\nError en la copia de elementos de hst_vector a dev_vector. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaMemcpy(dev_histograma,hst_histograma,M*sizeof(int),cudaMemcpyHostToDevice);
	//comprobamos la copia
	if(error != cudaSuccess){
		printf("\nError en la copia de elementos de hst_histograma a dev_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	threadperBlock = 512;
	blockperGrid = (N + threadperBlock-1)/threadperBlock;
	printf("Info: CUDA ejecutara %d hilos y %d bloques\n",threadperBlock,blockperGrid);

	//Ejecutaremos kernel y temporizadores
	printf("Info: Ejecutamos Kernel e iniciamos temporizadores\n");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//Iniciamos temporizador
	printf("Info: Temporizador iniciado\n");
	cudaEventRecord(start,0);
	histograma<<<blockperGrid,threadperBlock>>>(dev_histograma);
	kernel<<<blockperGrid,threadperBlock>>>(dev_vector,dev_histograma);
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
	error = cudaMemcpy(hst_histograma,dev_histograma,M*sizeof(int),cudaMemcpyDeviceToHost);
	//comprobamos la copia
	if(error != cudaSuccess){
		printf("\nError en la copia de elementos de dev_histograma a hst_histograma. (Code Error: %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	//comprobamos que la suma de los elementos del histograma corresponde con el numero de elementos
	for(int x=0;x<M;x++)
		sum_elements += hst_histograma[x];
	if(sum_elements != N){ //si no es igual, la ejecucion es incorrecta
		printf("Error, en el histograma hay %d elementos de 5 millones\n",sum_elements);
	}else{ //si es igual, la ejecucion es correcta y mostramos el histograma
		printf("\n\nInfo: Ejecución finalizada correctamente :) \n");
		mostrar_histograma(hst_histograma,sum_elements);
	}

	error = cudaFree(dev_vector);
	if(error != cudaSuccess){
		printf("Error, fallo librando el vector dev_vector (error code %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaFree(dev_histograma);
	if(error != cudaSuccess){
		printf("Error, fallo librando el vector dev_local_histograma (error code %s)\n",cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}


	return 0;
}

