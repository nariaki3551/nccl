/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "transport.h"

ncclResult_t ncclNetDeregister(struct ncclComm* comm, struct ncclReg* reg) {
  struct ncclRegCache* cache = &comm->regCache;
  ncclDebugNoWarn = NCCL_NET;
  for (int d=0; d<reg->nDevs; d++) {
    if (reg->handles[d] != NULL) NCCLCHECK(comm->ncclNet->deregMr(cache->sComms[reg->devs[d]], reg->handles[d]));
  }
  reg->nDevs = 0;
  free(reg->handles);
  reg->handles = NULL;
  ncclDebugNoWarn = 0;
  return ncclSuccess;
}

ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** reg) {
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;

  *reg = NULL;
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot]->addr) return ncclSuccess;
    if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      *reg = cache->slots[slot];
      return ncclSuccess;
    }
  }
}
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i=0; i<cache->population; i++) {
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)cache->slots[i]->addr, cache->slots[i]->pages);
    NCCLCHECK(ncclNetDeregister(comm, cache->slots[i]));
    // if (cache->slots[i]->state & NVLS_REG_COMPLETE) NCCLCHECK(ncclNvlsDeregBuffer(&cache->slots[i]->mcHandle, cache->slots[i]->regAddr, cache->slots[i]->dev, cache->slots[i]->regSize));
    free(cache->slots[i]);
  }
  free(cache->slots);
  for (int d=0; d<MAXCHANNELS; d++) {
    if (cache->sComms[d]) NCCLCHECK(comm->ncclNet->closeSend(cache->sComms[d]));
    if (cache->rComms[d]) NCCLCHECK(comm->ncclNet->closeRecv(cache->rComms[d]));
  }
  return ncclSuccess;
}
