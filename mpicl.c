#include <mpi.h> // MPI library for distributed computing
#include <stdio.h> // Standard I/O functions
#include <stdlib.h> // Standard library for memory allocation
#include <CL/cl.h> // OpenCL library

#define CHECK_ERROR(err) if (err != CL_SUCCESS) { printf("Error: %d\n", err); exit(1); }

// Kernel for matrix multiplication in OpenCL
const char *matrix_multiplication_kernel = 
    "__kernel void mat_mult(__global int* A, __global int* B, __global int* C, int n, int p) {"
    "    int row = get_global_id(0);"
    "    int col = get_global_id(1);"
    "    int result = 0;"
    "    for (int k = 0; k < n; k++) {"
    "        result += A[row * n + k] * B[k * p + col];"
    "    }"
    "    C[row * p + col] = result;"
    "}";

// Function to initialize matrices with random values
void initialize_matrices(int *A, int *B, int m, int n, int p) {
    for (int i = 0; i < m * n; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < n * p; i++) {
        B[i] = rand() % 10;
    }
}

int main(int argc, char *argv[]) {
    int rank, size; // MPI process rank and total number of processes
    int m = 4, n = 1000, p = 1000; // Matrix dimensions 

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes in the MPI communicator

    double start_time = MPI_Wtime(); // Record the start time for the execution

    int rows_per_proc = m / size; // Number of rows each process will handle

    // Allocate memory for local data in each process
    int *B = (int *)malloc(n * p * sizeof(int)); // Matrix B, common for all processes
    int *local_A = (int *)malloc(rows_per_proc * n * sizeof(int)); // Local rows of matrix A
    int *local_C = (int *)malloc(rows_per_proc * p * sizeof(int)); // Local rows of matrix C

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

    // OpenCL setup
    cl_int err;
    cl_uint platform_count;
    cl_platform_id platform_id;
    cl_device_id device_id;

    // Get the platform and device ID
    err = clGetPlatformIDs(1, &platform_id, &platform_count);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    CHECK_ERROR(err);

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    CHECK_ERROR(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERROR(err);

    // Create buffers for OpenCL
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rows_per_proc * n * sizeof(int), local_A, &err);
    CHECK_ERROR(err);

    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * p * sizeof(int), B, &err);
    CHECK_ERROR(err);

    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows_per_proc * p * sizeof(int), NULL, &err);
    CHECK_ERROR(err);

    // Create and build OpenCL program and kernel
    cl_program program = clCreateProgramWithSource(context, 1, &matrix_multiplication_kernel, NULL, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log if there's an error
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log: %s\n", log);
        free(log);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "mat_mult", &err);
    CHECK_ERROR(err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 4, sizeof(int), &p);
    CHECK_ERROR(err);

    // Set the global and local work sizes
    size_t global_work_size[] = { (size_t)rows_per_proc, (size_t)p };
    size_t local_work_size[] = { 16, 16 }; // Assuming optimal local work size

    // Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Read back the result to local_C
    err = clEnqueueReadBuffer(queue, buffer_C, CL_TRUE, 0, rows_per_proc * p * sizeof(int), local_C, 0, NULL, NULL);
    CHECK_ERROR(err);

    clFlush(queue);
    clFinish(queue);

    // Clean up OpenCL resources
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Gather the local results into matrix C at the root process
    MPI_Gather(local_C, rows_per_proc * p, MPI_INT, C, rows_per_proc * p, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // Record the end time for the execution

    if (rank == 0) { // Root process prints the total execution time
        printf("Total Execution Time: %.6f milliseconds\n", 1000 * (end_time - start_time)); // Convert seconds to milliseconds
        
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
