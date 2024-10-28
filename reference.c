#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


//********** Utilities *****************//
ncclResult_t setup(int *rank, int *device) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    if (strcmp(hostname, "snail01") == 0) {
      *rank = 0;
      *device = 0;
      if (setenv("NCCL_IB_HCA", "mlx5_1", 1) != 0) {
        return ncclSystemError;
      }
    } else if (strcmp(hostname, "snail02") == 0) {
      *rank = 1;
      *device = 0;
      if (setenv("NCCL_IB_HCA", "mlx5_2", 1) != 0) {
        return ncclSystemError;
      }
    } else {
      return ncclSystemError;
    }

    // Get device properties
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, *device);

    printf("Hostname: %s, Rank: %d, Device: %d, [0x%02x] %s\n", hostname, *rank, *device, prop.pciBusID, prop.name);

    char* hca = getenv("NCCL_IB_HCA");
    if (hca != NULL) {
        printf("Hostname: %s, NCCL_IB_HCA is set to: %s\n", hostname, hca);
    } else {
        printf("Hostname: %s, Failed to set NCCL_IB_HCA\n", hostname);
    }
    return ncclSuccess;
}

ncclUniqueId getUniqueId(int rank) {
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
        FILE *id_file = fopen("nccl_id.txt", "wb");
        fwrite(&id, sizeof(ncclUniqueId), 1, id_file);
        fclose(id_file);
    } else {
        sleep(1);
        FILE *id_file = fopen("nccl_id.txt", "rb");
        fread(&id, sizeof(ncclUniqueId), 1, id_file);
        fclose(id_file);
    }
    return id;
}
void ncclCheck(ncclResult_t result, const char* file, int line) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL error at %s:%d : %s\n", file, line, ncclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#define NCCLCHECK(call) ncclCheck((call), __FILE__, __LINE__)
//********** End of Utilities **********//


int main(int argc, char** argv) {
    int rank, device, nranks = 2;
    setup(&rank, &device);
    // Initialize CUDA
    cudaStream_t stream_ag, stream_rs;
    cudaStreamCreateWithFlags(&stream_ag, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_rs, cudaStreamNonBlocking);
    cudaSetDevice(device);

    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclUniqueId id = getUniqueId(rank);
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

    void *sendbuff_ag, *recvbuff_ag;
    void *sendbuff_rs, *recvbuff_rs;
    // int count = 128 * 1024 * 1024;
    int count = 2; // windowBytes = ??
    ncclDataType_t nccl_dtype = ncclFloat16;

    cudaMalloc(&sendbuff_ag, count * sizeof(nccl_dtype));
    cudaMalloc(&recvbuff_ag, count * nranks * sizeof(nccl_dtype));
    cudaMalloc(&sendbuff_rs, count * sizeof(nccl_dtype));
    cudaMalloc(&recvbuff_rs, count * nranks * sizeof(nccl_dtype));

    ncclGroupStart();
    ncclAllGather(sendbuff_ag, recvbuff_ag, count, nccl_dtype, comm, stream_ag);
    ncclReduceScatter(sendbuff_rs, recvbuff_rs, count, nccl_dtype, ncclSum, comm, stream_rs);
    ncclGroupEnd();

    cudaStreamSynchronize(stream_ag);
    cudaStreamSynchronize(stream_rs);

    ncclCommDestroy(comm);

    cudaFree(sendbuff_ag);
    cudaFree(recvbuff_ag);
    cudaFree(sendbuff_rs);
    cudaFree(recvbuff_rs);
    cudaStreamDestroy(stream_ag);
    cudaStreamDestroy(stream_rs);

    printf("Rank %d: Test finished successfully!\n", rank);
    return 0;
}

