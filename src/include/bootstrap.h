/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"

struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId), "Bootstrap handle is too large to fit inside NCCL unique ID");

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle);
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif
