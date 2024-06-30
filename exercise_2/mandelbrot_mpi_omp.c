#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

#include "pmg.h"

#define TAG_TASK_ROW_RESULT 9 
#define TAG_TASK_ROW 8
#define TAG_TASK_READY 7

int mandelbrot(double complex z, double complex c, int n, int max_iter){
    //int 
    n = 0;
    while(cabs(z) < 2.0 && n < max_iter) {
        z = z * z + c;
        n++;
    }
    return n >= max_iter ? 0 : n;
}

// Single thread Mandelbrot computation w/ OpenMP
unsigned short* mandelbrot_single_thread(int height, int width, double x_L, double x_R, double y_R, double y_L, double I_max){
    double d_x = (x_R - x_L) / (width - 1);
    double d_y = (y_R - y_L) / (height - 1);

    unsigned short *M = (unsigned short *)malloc(height * width * sizeof(unsigned short));

    #pragma omp parallel for schedule(dynamic)
    for(int a = 0; a < height * width; ++a){
        int i = a % width;
        int j = a / width;
        double x = x_L + i * d_x;
        double y = y_L + j * d_y;
        double complex c = x + y * I;
        M[a] = mandelbrot(0 * I, c, 0, I_max);
    }

    return M;
}

// Distributed Mandelbrot computation (rows) w/ OpenMP
unsigned short* mandelbrot_matrix_worker_row(int requested_row, int height, int width, double x_L, double x_R, double y_R, double y_L, double I_max){
    double d_x = (double) (x_R - x_L) / (double) (width - 1);
    double d_y = (double) (y_R - y_L) / (double) (height - 1);

    unsigned short *M = (unsigned short *)malloc(height * width * sizeof(unsigned short));

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < width; ++i){
        double x = x_L + i * d_x;
        double y = y_L + requested_row * d_y;
        double complex c = x + y * I;
        M[i] = mandelbrot(0 * I, c, 0, I_max);
    }

    return M;
}

// Distributed Mandelbrot computation (Master-Worker model)
// the master process (rank 0) distributes tasks (rows of the matrix) to worker processes using MPI communication
// Each worker process can use OpenMP to parallelize the computation of individual rows (they use the previous function in order to do so)
unsigned short* mandelbrot_matrix_master(int height, int width, double x_L, double x_R, double y_R, double y_L, double I_max){
    int rank, size, done=0;
    MPI_Status status;
    MPI_Request* recv_request;
    unsigned short *M;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the size of the MPI communicator

    if(rank>0){
        M = (unsigned short *)malloc(width * sizeof(unsigned short));
        MPI_Send(M, width, MPI_UNSIGNED_SHORT, 0, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD);
        printf("[rank %d] sent dummy result\n", rank);

        while(!done){
            recv_request = (MPI_Request*)malloc(sizeof(MPI_Request) * 2);
            int requested_row;

            // Setting the receival of the task that has to be performed
            MPI_Recv(&requested_row, 1, MPI_INT, 0, TAG_TASK_ROW, MPI_COMM_WORLD, &status);

            // Waiting for either the compilation message or the new task
            printf("[rank %d] waiting for either a new task or completion\n", rank);
            
            if(requested_row == -1){
                done = 1;
                printf("[rank %d] all done, terminated\n", rank);
            } 
            if(status.MPI_TAG == TAG_TASK_ROW && requested_row >= 0){
                printf("[rank %d] computing row %d\n", rank, requested_row);
                M = mandelbrot_matrix_worker_row(requested_row, height, width, x_L, x_R, y_R, y_L, I_max);
                printf("[rank %d] is sending back to master node\n", rank);
                MPI_Send(M, width, MPI_UNSIGNED_SHORT, 0, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD);
                free(recv_request);
                // free(M);
            }
        }
        free(M);
    } else {
        int n_row, next_row = 0, recv_row = 0;
        int* assigned_rows = (int*)malloc(sizeof(int*) * size);
        MPI_Status status;
        MPI_Request row_result_recv;
        int row_result_recvd;
        unsigned short* row;

        for(int i = 0; i < size; ++i){
            assigned_rows[i] = -1;
        }
        M = (unsigned short *)calloc(width * height, sizeof(unsigned short));
        row = (unsigned short *)malloc(width * sizeof(unsigned short));

        while(recv_row < height){
            MPI_Irecv(row, width, MPI_UNSIGNED, MPI_ANY_SOURCE, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD, &row_result_recv);

            MPI_Test(&row_result_recv, &row_result_recvd, &status);
            while(!row_result_recvd && next_row < height){
                n_row = next_row;

                row = mandelbrot_matrix_worker_row(next_row, height, width, x_L, x_R, y_R, y_L, I_max);
                memcpy(M + n_row * width, row, width * sizeof(unsigned short));
                next_row++;
                recv_row++;

                MPI_Test(&row_result_recv, &row_result_recvd, &status);
            }

            n_row = assigned_rows[status.MPI_SOURCE];
            if(n_row != -1){
                memcpy(M + n_row * width, row, width * sizeof(unsigned short));
                recv_row++;
            }

            if(next_row < height){
                printf("Assigning row %d to rank %d\n", next_row, status.MPI_SOURCE);
                MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK_ROW, MPI_COMM_WORLD);
                assigned_rows[status.MPI_SOURCE]=next_row;assigned_rows[status.MPI_SOURCE];
                next_row++;
            }
        }

        next_row = -1;
        for(int i = 0; i < size; ++i){
            MPI_Send(&next_row, 1, MPI_INT, i, TAG_TASK_ROW, MPI_COMM_WORLD);
        }

        free(assigned_rows);
        free(row);
        printf("All done\n");
    }

    return M;
}


int main(int argc, char *argv[]){

    if (argc < 9) {
        printf("Mandelbrot set: created with a mix of parallel and distributed implementation (based on OpenMP + MPI)\n");
        printf("How to use: %s width height x_L y_L x_R y_R I_max output_image\n", argv[0]);
        printf("Parameters description:\n");
        printf("Width: width of the output image\n");
        printf("Height: height of the output image\n");
        printf("x_L and y_L: components of the complex number c_L=x_L + i * y_L, bottom left corner of the considered portion of the complex plane\n");
        printf("x_R and y_R: components of the complex number c_R=x_R + i * y_R, top right corner of the considered portion of the complex plane\n");
        printf("I_max: iteration boundary before considering a candidate point to be part of the Mandelbrot set\n");
        printf("output_image: name of the image that will be generated\n");
        exit(1);
    }

    clock_t cputime;
    char* name_image;

    // Passing arguments form command line
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    double x_L = atof(argv[3]);
    double y_L = atof(argv[4]);
    double x_R = atof(argv[5]);
    double y_R = atof(argv[6]);
    int I_max = atoi(argv[7]);
    name_image = argv[8];

    // printf("Maximum dimension of M = %d\n", max_M);
    // if(I_max>max_M){
    //     printf("Error: the image is to large (%d > %d)\n", I_max, max_M);
    //     exit(1);
    // }

    /* We need both OpenMPI and MPI to run properly: check for the OMPI_COMM_WORLD_SIZE 
     environment variable, which is set by mpirun on start. */
    char* world_size;
    world_size = getenv("OMPI_COMM_WORLD_SIZE");

    if(world_size == NULL){
        printf("Error: It seems that the program is not able to run with mpirun. Plese run it using mpi [options] %s\n", argv[0]);
        exit(1);
    }
    
    /* Initialize MPI using the MPI_THREAD_FUNNELED threading option, which allows only
     the master thread in every process (rank) to perform MPI calls. */
    int mpi_thread_init;
    int rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_init);
    if(mpi_thread_init<MPI_THREAD_FUNNELED){
        printf("Error: Iniziatilation of MPI with MPI_THREAD_FUNNELED cannot be performed");
        exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // 65535 for int or 255 for char
    unsigned short *M;
    if(size == 1){
        M = mandelbrot_single_thread(height, width, x_L, x_R, y_R, y_L, I_max);
    } else {
        M = mandelbrot_matrix_master(height, width, x_L, x_R, y_R, y_L, I_max);
    }    

    cputime = clock();
    double walltime = (double) cputime / CLOCKS_PER_SEC;
    printf("[rank %d] cpu time: %f seconds\n", rank, walltime);

    if (rank == 0) {
        // Matrix M written in a pmg file
        write_pgm_image(M, I_max, height, width, "mandelbrot.pgm");
    }

    free(M);
    MPI_Finalize();
    return 0;
}