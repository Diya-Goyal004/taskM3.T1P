#include <mpi.h> // Include the MPI library for distributed computing
#include <stdio.h> // Standard input/output functions
#include <stdlib.h> // Standard library for memory allocation and random numbers

// Function to initialize matrices with random values
void initialize_matrices(int *A, int *B, int m, int n, int p) {
    // Fill matrix A with random values
    for (int i = 0; i < m * n; i++) {
        A[i] = rand() % 10; // Random value between 0 and 9
    }
    // Fill matrix B with random values
    for (int i = 0; i < n * p; i++) {
        B[i] = rand() % 10; // Random value between 0 and 9
    }
}

// Function to multiply a row of A with the entire matrix B
void multiply_row_with_B(int *row, int *B, int *result, int n, int p) {
    // Multiply a row of matrix A with matrix B to produce a row of matrix C
    for (int j = 0; j < p; j++) {
        result[j] = 0; // Initialize the result to zero
        // Perform the dot product between the row and the corresponding column in B
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
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes in the MPI communicator

    double start_time = MPI_Wtime(); // Record the start time for the execution

    // Ensure that the matrix dimensions can be evenly divided among processes
    int rows_per_proc = m / size; // Number of rows each process will handle

    // Allocate memory for the local data in each process
    int *B = (int *)malloc(n * p * sizeof(int)); // Allocate memory for matrix B (common for all processes)
    int *local_A = (int *)malloc(rows_per_proc * n * sizeof(int)); // Memory for local rows of matrix A
    int *local_C = (int *)malloc(rows_per_proc * p * sizeof(int)); // Memory for local rows of matrix C

    int *A = NULL; // Initialize A to NULL; only root will allocate memory for it
    int *C = NULL; // Initialize C to NULL; only root will allocate memory for it

    if (rank == 0) { // Root process initializes matrices
        A = (int *)malloc(m * n * sizeof(int)); // Allocate memory for full matrix A
        C = (int *)malloc(m * p * sizeof(int)); // Allocate memory for full matrix C
        initialize_matrices(A, B, m, n, p); // Initialize matrices A and B with random values
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * p, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast B from root to all processes

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, rows_per_proc * n, MPI_INT, local_A, rows_per_proc * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply local rows with B to get local_C
    for (int i = 0; i < rows_per_proc; i++) {
        multiply_row_with_B(&local_A[i * n], B, &local_C[i * p], n, p); // Multiply local rows
    }

    // Gather the local results into matrix C at the root process
    MPI_Gather(local_C, rows_per_proc * p, MPI_INT, C, rows_per_proc * p, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // Record the end time for the execution

    if (rank == 0) { // Root process prints the total execution time
        printf("Total Execution Time: %.6f milliseconds\n", 1000 * (end_time - start_time)); // Convert seconds to milliseconds
        
        // Free the memory allocated for A and C
        free(A);
        free(C);
    }

    // Free the memory for B, local_A, and local_C
    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize(); // Finalize MPI

    return 0;
}
