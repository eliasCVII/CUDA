/*****************************************************
 * Nombre del archivo: MPI-CUDA.cu
 * Programmer: Elias Contreras V, Martin Gomez J
 * Santiago de Chile, 28-2-2026
 ****************************************************/

#include "/usr/include/mpich/mpi.h"
// #include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MASTER 0
#define TAG_0 0
#define TAG_1 1
#define TAG_2 2
#define VERBOSE "-V"
#define SILENT "-S"

/*
 *
 */
void Usage(char *arg) {

  printf("\nUsage: %s k [-V | -S] < data.txt\n", arg);
  fflush(stdout);
}

/*
 *
 */
float *allocate_matrix(int rows, int cols) {

  return (float *)calloc(rows * cols, sizeof(float));
}

/*
 *
 */
void print_matrix(float *matrix, int rows, int cols) {
  int i, j;

  for (i = 0; i < rows; i = i + 1) {
    for (j = 0; j < cols; j = j + 1) {
      printf("%.1f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("-----------------------------\n");
  fflush(stdout);
}

/*
 *
 */
__global__ void matrix_multiplication_kernel(float *A, float *B, float *C,
                                             int f, int c1, int c2) {
  int row, col, k;
  float sum;

  row = blockIdx.y * blockDim.y + threadIdx.y;
  col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < f) {
    if (col < c2) {
      sum = 0.0;
      for (k = 0; k < c1; k = k + 1) {
        sum = sum + A[row * c1 + k] * B[k * c2 + col];
      }
      C[row * c2 + col] = sum;
    }
  }
}

/*
 *
 */
int main(int argc, char *argv[]) {
  int rank, size, n, k_threads, tasks_sent, workers, f, c1, c2, i, j, source,
      task_count, *mensaje;
  float *A, *B, *C, *d_A, *d_B, *d_C;
  char *modo;
  MPI_Status status;
  time_t total_ts, total_te;
  clock_t cs, ce;
  dim3 block, grid;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (argc < 3) {
    if (rank == MASTER) {
      Usage(argv[0]);
    }
    MPI_Finalize();
    return 0;
  }
  k_threads = atoi(argv[1]);
  modo = argv[2];
  mensaje = (int *)calloc(3, sizeof(int));
  if (rank == MASTER) {
    if (scanf("%d", &n) != 1) {
      n = 0;
    }
    total_ts = time(NULL);
    tasks_sent = 0;
    workers = size - 1;
    if (strcmp(modo, SILENT) == 0) {
      printf("%d Nodos participantes\n", size);
    }
    for (i = 1; i < size; i = i + 1) {
      if (tasks_sent < n) {
        if (scanf("%d %d %d", &f, &c1, &c2) != 3) {
          f = 0;
          c1 = 0;
          c2 = 0;
        }
        mensaje[0] = f;
        mensaje[1] = c1;
        mensaje[2] = c2;
        MPI_Send(mensaje, 3, MPI_INT, i, TAG_1, MPI_COMM_WORLD);
        tasks_sent = tasks_sent + 1;
      } else {
        MPI_Send(NULL, 0, MPI_INT, i, TAG_0, MPI_COMM_WORLD);
        workers = workers - 1;
      }
    }
    while (workers > 0) {
      MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, TAG_2, MPI_COMM_WORLD,
               &status);
      source = status.MPI_SOURCE;
      if (tasks_sent < n) {
        if (scanf("%d %d %d", &f, &c1, &c2) != 3) {
          f = 0;
          c1 = 0;
          c2 = 0;
        }
        mensaje[0] = f;
        mensaje[1] = c1;
        mensaje[2] = c2;
        MPI_Send(mensaje, 3, MPI_INT, source, TAG_1, MPI_COMM_WORLD);
        tasks_sent = tasks_sent + 1;
      } else {
        MPI_Send(NULL, 0, MPI_INT, source, TAG_0, MPI_COMM_WORLD);
        workers = workers - 1;
      }
    }
    total_te = time(NULL);
    if (strcmp(modo, SILENT) == 0) {
      printf("Total Wall Time: %ld s\n", (long)(total_te - total_ts));
    }
  } else {
    task_count = 1;
    MPI_Recv(mensaje, 3, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    while (status.MPI_TAG != TAG_0) {
      f = mensaje[0];
      c1 = mensaje[1];
      c2 = mensaje[2];
      A = allocate_matrix(f, c1);
      B = allocate_matrix(c1, c2);
      C = allocate_matrix(f, c2);
      for (i = 0; i < f; i = i + 1) {
        for (j = 0; j < c1; j = j + 1) {
          A[i * c1 + j] = 1.0;
        }
      }
      for (i = 0; i < c1; i = i + 1) {
        for (j = 0; j < c2; j = j + 1) {
          B[i * c2 + j] = 2.0;
        }
      }
      if (strcmp(modo, VERBOSE) == 0) {
        printf("[Node %d | Job %d]: %dx%d * %dx%d\n", rank, task_count, f, c1,
               c1, c2);
        printf("A [Node %d | Job %d]:\n", rank, task_count);
        print_matrix(A, f, c1);
        printf("B [Node %d | Job %d]:\n", rank, task_count);
        print_matrix(B, c1, c2);
      }
      cudaMalloc((void **)&d_A, f * c1 * sizeof(float));
      cudaMalloc((void **)&d_B, c1 * c2 * sizeof(float));
      cudaMalloc((void **)&d_C, f * c2 * sizeof(float));
      cudaMemcpy(d_A, A, f * c1 * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, c1 * c2 * sizeof(float), cudaMemcpyHostToDevice);
      if (k_threads > 1024) {
        block.x = 32;
        block.y = 32;
      } else {
        block.x = k_threads;
        block.y = 1;
      }
      block.z = 1;
      grid.x = (c2 + block.x - 1) / block.x;
      grid.y = (f + block.y - 1) / block.y;
      grid.z = 1;
      cs = clock();
      matrix_multiplication_kernel<<<grid, block>>>(d_A, d_B, d_C, f, c1, c2);
      cudaDeviceSynchronize();
      ce = clock();
      cudaMemcpy(C, d_C, f * c2 * sizeof(float), cudaMemcpyDeviceToHost);
      if (strcmp(modo, VERBOSE) == 0) {
        printf("C [Node %d | Job %d]:\n", rank, task_count);
        print_matrix(C, f, c2);
      }
      if (strcmp(modo, SILENT) == 0) {
        printf("[Node %d]: %dx%d. CPU Time: %f s\n", rank, f, c2,
               (float)(ce - cs) / CLOCKS_PER_SEC);
      }
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      free(A);
      free(B);
      free(C);
      task_count = task_count + 1;
      MPI_Send(NULL, 0, MPI_INT, MASTER, TAG_2, MPI_COMM_WORLD);
      MPI_Recv(mensaje, 3, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status);
    }
  }
  free(mensaje);
  MPI_Finalize();
  return 0;
}
