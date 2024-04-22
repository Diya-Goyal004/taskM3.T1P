#include <mpi.h> // MPI library
#include <stdio.h> // Standard I/O
#include <stdlib.h> // Standard library for memory allocation and random numbers
#include <omp.h> // OpenMP library

// Initialize matrices with random values
void initialize_matrices(int *A, int *B, int m, int n, int p) {
    for (int i = 0; i < m * n; i++) {
        A[i] = rand() % 10; // Random value between 0 and 9
    }
    for (int i = 0; i < n * p; i++) {
        B[i] = rand() % 10; // Random value between 0 and 9
    }
}

// Multiply a row of A with the entire matrix B
void multiply_row_with_B(int *row, int *B, int *result, int n, int p) {
    for (int j = 0; j < p; j++) {
        result[j] = 0; // Initialize result to zero
        for (int k = 0; k < n; k++) {
            result[j] += row[k] * B[k * p + j]; // Matrix multiplication
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size; // MPI process rank and total number of processes
    int m = 4, n = 1000, p = 1000; // Matrix dimensions

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    double start_time = MPI_Wtime(); // Start time

    int rows_per_proc = m / size; // Number of rows each process will handle

    // Allocate memory for local data
    int *B = (int *)malloc(n * p * sizeof(int)); // Common matrix B for all processes
    int *local_A = (int *)malloc(rows_per_proc * n * sizeof(int)); // Local rows of A
    int *local_C = (int *)malloc(rows_per_proc * p * sizeof(int)); // Local rows of C

    int *A = NULL;
    int *C = NULL;

    if (rank == 0) { // Root process initializes matrices
        A = (int *)malloc(m * n * sizeof(int)); // Full matrix A
        C = (int *)malloc(m * p * sizeof(int)); // Full matrix C
        initialize_matrices(A, B, m, n, p); // Initialize A and B
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * p, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast B from root

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, rows_per_proc * n, MPI_INT, local_A, rows_per_proc * n, MPI_INT, 0, MPI_COMM_WORLD); // Scatter A

    // Multiply local rows with B in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < rows_per_proc; i++) {
        multiply_row_with_B(&local_A[i * n], B, &local_C[i * p], n, p); // Parallel row multiplication
    }

    // Gather the results back to the root process
    MPI_Gather(local_C, rows_per_proc * p, MPI_INT, C, rows_per_proc * p, MPI_INT, 0, MPI_COMM_WORLD); // Gather results

    double end_time = MPI_Wtime(); // End time

    if (rank == 0) { // Root process prints the total execution time
        printf("Total Execution Time: %.6f milliseconds\n", 1000 * (end_time - start_time)); // Output time in ms
        
        // Free memory for A and C
        free(A);
        free(C);
    }

    // Free memory for B, local_A, and local_C
    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize(); // Finalize MPI
    return 0;
}
