/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "channel.h"
#include "cudawrap.h"
#include "profiler.h"
#include "transport.h"

#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

// Returns maximum kernel stack size of all CUDA kernels
ncclResult_t ncclInitKernelsForDevice(int cudaArch, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();

  for (int k=0; k < ncclDevKernelCount; k++) {
    void* fn = ncclDevKernelList[k];
    if (fn == nullptr) continue;

    if (maxStackSize) {
      cudaFuncAttributes attr = {0};
      CUDACHECKGOTO(cudaFuncGetAttributes(&attr, fn), result, ignore0);
      if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
    ignore0:;
    }
    if (carveout) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
        result, ignore1);
    ignore1:;
    }
    if (ncclShmemDynamicSize(cudaArch) != 0) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, ncclShmemDynamicSize(cudaArch)),
        result, next_kernel);
    }
  next_kernel:;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Data movement metrics.

static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce: return 2;
  case ncclFuncAllGather: return nRanks;
  case ncclFuncReduceScatter: return nRanks;
  default: return 1;
  }
}
static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks*count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks*count : count;
}
static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks*count : count;
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

static ncclResult_t addProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

static void addWorkBatchToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset,
    int p2pRound = -1
  ) {
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  size_t workSize = ncclDevWorkSize(workType);
  // Conditions causing us to create a new blank batch.
  bool newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    // All of the conditions that prevent us from appending to current batch.
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // The following ensure the device can handle a batch this large. They have to
    // account for all extension batches being fused together which is why
    // wipBatch.workBytes and wipBatch.nP2ps aren't reset to 0 for a new extension
    // batch further down.
    newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    if (workType == ncclDevWorkTypeP2p) {
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      for (int i=0; i < chan->wipBatch.nP2ps; i++) {
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
      }
    }
  }
  // Conditions causing us to create an extension batch (prev->nextExtends=1)
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  bool extendBatch = 63*workSize < offset;
  extendBatch |= 0 != offset%workSize;
  if (newBatch || extendBatch) {
    if (!newBatch) batch->nextExtends = extendBatch; // Extending the previous batch.
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity thinks that ncclIntruQueueEnqueue will access chan->workBatchQueue->tail, which might
    // be NULL.  But that code is guarded by chan->workBatchQueue->head not being NULL, in which
    // case tail won't be NULL either.
    // coverity[var_deref_model:FALSE]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->nextExtends = 0;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
    offset = 0;
    if (newBatch) {
      // Since extension batches are fused together on the device, and these values
      // account for constraints on the fused batch, we only reset the values on
      // a new batch
      chan->wipBatch.workBytes = 0;
      chan->wipBatch.nP2ps = 0;
      // We don't count extension batches since this is used to derive a proxyOpCount,
      // and we wan't all ops which are fused together to have the same value.
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
    }
    plan->nWorkBatches += 1;
  }
  batch->offsetBitset |= 1ull<<(offset/workSize);
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {
    // We need to ensure that a single batch doesn't have multiple p2p's
    // of the same round since they would use the same connections.
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound;
  }
}

static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // If we can fit everything into the kernel args we do so.
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;

  // Put batches into the kernel arguments. The first batch for each channel
  // must be located at batchZero[blockIdx.x]. To achieve this we round robin
  // over the channels in ascending order until they're exhausted.
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);
  }

  // Merge-sort per-channel proxy-op lists by opCount when merging them into plan->proxyOpQueue
  // Phase 1: scan first op of each channel, store opCount in headIds[c].
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c+1;
  }
  // Phase 2: Dequeue from planner->channels[c], enqueue in merged order to plan
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // Find channel with least proxy-op id. We store the heads[c]->opCount in
    // headIds[c] to remove indirect loads from this loop.
    for (int c1=0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { c = c1; minId = id; }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
}

int64_t ncclParamLocalRegister();
NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  void* ptr;
};
static ncclResult_t cleanupIpc(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclIpcCleanupCallback* me = (struct ncclIpcCleanupCallback*)cb;
  CUDACHECKIGNORE(cudaIpcCloseMemHandle(me->ptr));
  free(me);
  return ncclSuccess;
}

static ncclResult_t registerCollBuffers(
    struct ncclComm* comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue,
    bool* regNeedConnect
  ) {
  ncclResult_t result = ncclSuccess;

  info->regBufType = NCCL_REGULAR_BUFFER;
  *regNeedConnect = true;
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
#if CUDART_VERSION >= 11030
   if ((info->algorithm == NCCL_ALGO_COLLNET_DIRECT || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) && comm->collNetRegSupport) {
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    // size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);
    int sendRegBufFlag = 0;
    int recvRegBufFlag = 0;
    void *sendHandle; //, *recvHandle;

    if (ncclParamLocalRegister()) {
      ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &sendRegBufFlag, &sendHandle);
      info->sendMhandle = sendHandle;
      if (sendRegBufFlag) {
        // ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &recvRegBufFlag, &recvHandle);
        // info->recvMhandle = recvHandle;
      }
    }

    if ((sendRegBufFlag == 0 || recvRegBufFlag == 0) && comm->planner.persistent && ncclParamGraphRegister()) {
    }

    if (sendRegBufFlag && recvRegBufFlag) {
    }
  }
exit:
#endif
  return result;
}

static ncclResult_t getCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* task, int* collNetSupport);
static ncclResult_t getAlgoInfo(
  struct ncclComm* comm, struct ncclTaskColl* task,
  int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo = NULL
);
static ncclResult_t calcCollChunking(
  struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
  /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
);

struct ncclKernelPlanBudget {
  ssize_t inArgsBytes; // Space available within kernel args struct
  ssize_t outArgsBytes; // Space available outside of args struct (fifo or persistent buf)
};

static bool testBudget(
    struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes
  ) {
  ssize_t batchBytes = nWorkBatches*sizeof(struct ncclDevWorkBatch);
  bool ok = false;
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;
}

// Called once per ncclGroup to organize the user submitted tasks in
// comm->planner so that they can be peeled off into plans.
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // Tasks from the sorter come out ordered size descending.
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // Tasks are assembled by (fn,op,ty) size ascending.
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));
  int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  int fnOpTyCount = 0;

  // Walk the size sorted tasks, binning them by (fn,op,ty).
  while (task != nullptr) {
    struct ncclTaskColl* next = task->next;
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    // Add to set of (fn,op,ty) indices on first occurrence
    if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
    // Add to LIFO for this (fn,op,ty)
    task->next = tasksByFnOpTy[index];
    tasksByFnOpTy[index] = task;
    // Next task
    task = next;
  }

  // Walk (fn,op,ty) bins, compute algo and proto etc. Then bin them by their
  // scheduling constraints (collnet x nvls).
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
  for (int cursor=0; cursor < fnOpTyCount; cursor++) {
    struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]];
    int collNetSupport = 0;
    NCCLCHECK(getCollNetSupport(comm, aggBeg, &collNetSupport));
    int nvlsSupport = false;
    // Crudely estimate number of tasks per channel. This is using the wrong number
    // of channels for NVLS algos, but knowing the algo requires having this value,
    // so either be crude our iterate until fixed point, we chose the former.
    int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);
    do {
      struct ncclTaskColl* aggEnd = aggBeg->next;
      struct ncclTaskColl agg = *aggBeg;
      // We aggregate operations that are within 4X size of each other.
      while (aggEnd != nullptr && aggEnd->trafficBytes < 4*aggBeg->trafficBytes) {
        agg.count += aggEnd->count;
        agg.trafficBytes += aggEnd->trafficBytes;
        aggEnd = aggEnd->next;
      }

      NCCLCHECK(getAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
      agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);

      int isCollnet=0, isNvls=0;
      switch (agg.algorithm) {
      case NCCL_ALGO_NVLS:
      case NCCL_ALGO_NVLS_TREE:
        isNvls = 1;
        isCollnet = agg.algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1;
        break;
      case NCCL_ALGO_COLLNET_CHAIN:
      case NCCL_ALGO_COLLNET_DIRECT:
        isCollnet = 1;
        break;
      }
      // Update the aggregated tasks with the computed values.
      do {
        struct ncclTaskColl* next = aggBeg->next;
        aggBeg->algorithm = agg.algorithm;
        aggBeg->protocol = agg.protocol;
        aggBeg->nMaxChannels = agg.nMaxChannels;
        aggBeg->nWarps = agg.nWarps;
        aggBeg->devFuncId = agg.devFuncId;
        aggBeg->isCollnet = isCollnet;
        aggBeg->isNvls = isNvls;
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
        aggBeg = next;
      } while (aggBeg != aggEnd);
    } while (aggBeg != nullptr);
  }

  // Concatenate `collBins[*][*]` together into final list `planner->collTaskQueue`.
  // Collnet is the outer dimension since that affects how we divide over the
  // channels.
  for (int isCollnet=0; isCollnet <= 1; isCollnet++) {
    for (int isNvls=0; isNvls <= 1; isNvls++) {
      ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
    }
  }

  // Walk tasks again to:
  // 1. Possibly register buffers.
  // 2. Build ncclDevWorkColl structs.
  // 3. Bin the work structs according to the number of valid channels they
  //    may be assigned to {collnet, nvls, standard}
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // Build a ncclDevWorkColl[Reg?] struct for each task.
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    registerCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      if (task->algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false && regNeedConnect == true) {
        comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
        algoNeedConnect[NCCL_ALGO_NVLS] = true;
      }
      if (task->algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    }

    struct ncclDevWorkColl devWork = {};
    devWork.sendbuff = (void*)task->sendbuff;
    devWork.recvbuff = (void*)task->recvbuff;
    devWork.sendbuffOffset = task->sendbuffOffset;
    devWork.recvbuffOffset = task->recvbuffOffset;
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    devWork.root = task->root;
    devWork.nWarps = task->nWarps;
    devWork.redOpArg = task->opDev.scalarArg;
    devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
    devWork.oneNode = (comm->nNodes == 1);
    devWork.regUsed = task->regBufType;

    struct ncclWorkList* workNode;
    switch (task->regBufType) {
    case NCCL_REGULAR_BUFFER:
    case NCCL_IPC_REG_BUFFER:
    case NCCL_COLLNET_REG_BUFFER:
      { workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeColl;
        workNode->size = sizeof(struct ncclDevWorkColl);
        memcpy((void*)(workNode+1), (void*)&devWork, workNode->size);
      } break;
    case NCCL_NVLS_REG_BUFFER:
      { struct ncclDevWorkCollReg workReg = {};
        workReg.coll = devWork; // C++ struct assignment
        /* NVLS only has one send and recv buffer registered */
        workReg.dnInputs[0] = regBufSend[0];
        workReg.dnOutputs[0] = regBufRecv[0];
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeCollReg;
        workNode->size = sizeof(struct ncclDevWorkCollReg);
        memcpy((void*)(workNode+1), (void*)&workReg, workNode->size);
      } break;
    default:
      /* impossible value */
      WARN("Invalid regBufType %d", task->regBufType);
      return ncclInvalidArgument;
    }

    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
    task = task->next;
  }

  return ncclSuccess;
}

static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
  ) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // Estimate number of tasks that will fit in this plan.
  int nPlanColls = 0;
  size_t trafficBytes[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int nChannels[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int const nMaxChannels[2*2] = {comm->nChannels, -1, // [collnet][nvls]
                                 comm->nChannels, -1};
  constexpr size_t MinTrafficPerChannel = 16 << 10; // 16K traffic as minimal
  do {
    size_t workBytes = 0;
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    while (task != nullptr) {
      int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
      if (!testBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;

      nPlanColls += 1;
      workBytes += workNode->size;
      int kind = 2*task->isCollnet + task->isNvls;
      trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);
      nChannels[kind] += task->nMaxChannels;
      nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);
      task = task->next;
      workNode = workNode->next;
    }
  plan_full:;
  } while (0);

  int kindPrev = -1;
  size_t trafficPerChannel = 0;
  int channelId = 0;
  size_t currentTraffic = 0;
  while (nPlanColls!=0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1);
    size_t elementSize = ncclTypeSize(task->datatype);

    int kind = 2*task->isCollnet + task->isNvls;
    if (kind != kindPrev) {
      trafficPerChannel = std::max<size_t>(MinTrafficPerChannel, trafficBytes[kind]/nChannels[kind]);
      kindPrev = kind;
      channelId = 0;
      currentTraffic = 0;
    }

    if (task->isCollnet) {
      int nChannels = task->nMaxChannels;
      // Ensure room for worst case of one new batch per channel
      if (!testBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOp;
      uint32_t chunkSize, directFlags=0;
      NCCLCHECK(calcCollChunking(comm, task, nChannels, globalBytesPerElement*task->count, &chunkSize, &directFlags, &proxyOp));
      devWork->channelLo = 0;
      devWork->channelHi = nChannels-1;
      devWork->collnet.count = task->count;
      devWork->collnet.chunkCount = chunkSize/ncclTypeSize(task->datatype);
      devWork->direct = directFlags;

      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        proxyOp.channelId = c;
        proxyOp.opCount = proxyOpId;
        proxyOp.task.coll = task;
        proxyOp.rank = comm->rank;
        addWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        NCCLCHECK(addProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    }

    plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);
    plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps*WARP_SIZE);
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
    }

    if (comm->rank == 0) {
      if (task->isCollnet) {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count=%ld chunkCount=%d",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->collnet.count, devWork->collnet.chunkCount);
      } else {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
          int(devWork->cbd.chunkGrainsLo*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsMid*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsHi*ncclProtoGrainSize(task->protocol)));
      }
    }

    for (int i=0; i < task->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);
    nPlanColls -= 1;
    planner->nTasksColl -= 1;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

static int calcP2pChannelCount(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = divUp(totalSize, nChannels);
  }
  return nChannels;
}

// static ncclResult_t scheduleP2pTasksToPlan(
//     struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
//   ) {
//   int nRanks = comm->nRanks;
//   struct ncclKernelPlanner::Peer* peers = comm->planner.peers;

//   plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
//   if (!plan->kernelSpecialized) {
//     plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
//     plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
//   }

//   // Compute how much to split operations
//   // Try to use all channels
//   int nChannelsMax = comm->p2pnChannelsPerPeer;
//   int nChannelsMin = nChannelsMax;
//   // Try to use all channels, but one channel per operation.
//   while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

//   while (comm->planner.nTasksP2p != 0) {
//     for (int round=0; round < nRanks; round++) {
//       int sendRank = comm->p2pSchedule[round].sendRank;
//       int recvRank = comm->p2pSchedule[round].recvRank;
//       struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
//       struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
//       if (send == nullptr && recv == nullptr) continue;

//       if (sendRank == comm->rank) {
//         if (send != nullptr && recv == nullptr) {
//           WARN("Trying to send to self without a matching recv");
//           return ncclInvalidUsage;
//         }
//         if (send == nullptr && recv != nullptr) {
//           WARN("Trying to recv to self without a matching send");
//           return ncclInvalidUsage;
//         }
//       }
//       ssize_t sendBytes = send ? send->bytes : -1;
//       ssize_t recvBytes = recv ? recv->bytes : -1;
//       void* sendBuff = send ? send->buff : nullptr;
//       void* recvBuff = recv ? recv->buff : nullptr;

//       if (sendRank == comm->rank && send->buff == recv->buff) {
//         // Skip send to self in-place (we don't need to support this).
//         ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
//         ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
//         comm->planner.nTasksP2p -= 2;
//       } else {
//         // Ensure room for worst case of one new batch per channel.
//         if (!testBudget(budget, plan->nWorkBatches+nChannelsMax, plan->workBytes + sizeof(struct ncclDevWorkP2p))) {
//           return ncclSuccess;
//         }
//         struct ncclTaskP2p* p2pTasks[2] = { recv, send };
//         NCCLCHECK(addP2pToPlan(comm, plan, nChannelsMin, nChannelsMax, round, sendRank, sendBuff, sendBytes, recvRank, recvBuff, recvBytes, p2pTasks));
//         if (send != nullptr) {
//           ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
//           ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);
//           comm->planner.nTasksP2p -= 1;
//         }
//         if (recv != nullptr) {
//           ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
//           ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);
//           comm->planner.nTasksP2p -= 1;
//         }
//       }
//     }
//   }
//   return ncclSuccess;
// }

namespace {
  struct uploadWork_cleanup_t {
    struct ncclCommEventCallback base;
    void *hostBuf;
  };
  ncclResult_t uploadWork_cleanup_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
    free(me->hostBuf);
    CUDACHECK(cudaEventDestroy(me->base.event));
    return ncclSuccess;
  }
}

static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  // case ncclDevWorkStorageTypeFifo:
  //   fifoBufHost = comm->workFifoBuf;
  //   fifoCursor = comm->workFifoProduced;
  //   fifoMask = comm->workFifoBytes-1;
  //   waitWorkFifoAvailable(comm, fifoCursor + workBytes);
  //   plan->kernelArgs->workBuf = comm->workFifoBufDev;
  //   break;
  // case ncclDevWorkStorageTypePersistent:
  //   static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
  //   fifoBufHost = malloc(workBytes);
  //   fifoCursor = 0;
  //   fifoMask = ~0u;
  //   break;
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  for (int b=0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // Write the channel-shared work structs.
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode+1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16),
        __builtin_assume_aligned(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    comm->workFifoProduced = fifoCursor;
    if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    break;
  case ncclDevWorkStorageTypePersistent:
    { ncclResult_t result = ncclSuccess;
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      void* fifoBufDev = nullptr;
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

      // Acquire deviceStream to gain access to deviceStream.cudaStream. Since the
      // user's graph will be launched later, and it also acquires the deviceStream,
      // it will observe this upload.
      NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream), result, finish_scope);

      CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);
      plan->workBufPersistent = fifoBufDev;
      plan->kernelArgs->workBuf = fifoBufDev;

      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);
      cudaEvent_t memcpyDone;
      CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, finish_scope);
      CUDACHECKGOTO(cudaEventRecord(memcpyDone, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);

      struct uploadWork_cleanup_t* cleanup;
      NCCLCHECK(ncclCalloc(&cleanup, 1));
      cleanup->base.fn = uploadWork_cleanup_fn;
      cleanup->base.event = memcpyDone;
      cleanup->hostBuf = fifoBufHost;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cleanup->base);

      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), result, finish_scope);
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm), result, finish_scope);

    finish_scope:
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      if (result != ncclSuccess) return result;
    } break;
  default: break;
  }
  return ncclSuccess;
}

static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // Advance comm's collOpCount by number of colls in this plan.
  comm->sharedRes->collOpCount += plan->collOpCount;

  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    op->profilerContext = comm->profilerContext;
    op->eActivationMask = op->coll <= ncclFuncAllReduce ? op->task.coll->eActivationMask : op->task.p2p->eActivationMask;
    op->taskEventHandle = op->coll <= ncclFuncAllReduce ? op->task.coll->eventHandle : op->task.p2p->eventHandle;
    ncclProfilerAddPidToProxyOp(op);

    uint64_t oldId = op->opCount;
    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    if (oldId & 1) { // p2p
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      p2pOpBump[op->channelId] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId]<<1) + oldId;
    } else { // coll
      op->opCount = (collOpCount<<1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    op->opCount = oldId; // Restore for next uploadProxyOps()

    struct ncclProxyOp* opNext = op->enqNext;
    if (!plan->persistent) {
      // Non-persistent kernels upload ops only once so can be free'd here.
      ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, op);
    }
    op = opNext;
  }

  // Erase proxyOpQueue since all ops were free'd back to mempool.
  if (!plan->persistent) ncclIntruQueueConstruct(&plan->proxyOpQueue);

  for (int c=0; c < MAXCHANNELS; c++) {
    // Advance channel's p2pOpCount by number of p2p's in this plan channel.
    comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  NCCLCHECK(uploadProxyOps(comm, plan));
  NCCLCHECK(ncclProxyStart(comm));
  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim plan concurrently.
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->persistentRefs -= 1;
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      CUDACHECK(cudaFree(plan->workBufPersistent));
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    }
    struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
    while (q != nullptr) {
      struct ncclProxyOp* q1 = q->enqNext;
      ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
      q = q1;
    }
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    while (ct != nullptr) {
      struct ncclTaskColl* ct1 = ct->next;
      ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
      ct = ct1;
    }
    ncclResult_t result = ncclSuccess;
    while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
      ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
      if (res1 != ncclSuccess) result = res1;
    }
    NCCLCHECK(result);
  }
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  planner->persistent = persistent;
  int nPlans = 0;

  // Poll for callbacks sent to us from other threads. Typically these free
  // resources from to our memory pools.
  NCCLCHECK(ncclCommPollCallbacks(comm, /*waitSome=*/false));

  if (planner->nTasksColl != 0) {
    do {
      memset(&planner->wipPlan, 0, sizeof(planner->wipPlan));

      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan;
      plan->persistent = persistent;
      // finishPlan() promotes ncclDevWorkStorageType[Fifo|Persistent]->Args if the work can fit.
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent
                                         : ncclDevWorkStorageTypeFifo;

      struct ncclKernelPlanBudget budget;
      budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
      // Non-persistent kernels fill up at most half of our fifo per kernel.
      budget.outArgsBytes = plan->persistent ? (1<<30) : comm->workFifoBytes/2;

      // Drain coll tasks first. This is essential since we partition tasks based
      // on the work budget and p2p work isn't collective. If we were to drain p2p
      // first, the place where we cut the kernel could vary by rank which would
      // cause the "shortest channel first" channel picker to have divergent results.
      if (planner->nTasksColl != 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
      }
      // // And only drain p2p tasks once colls are depleted.
      // if (planner->nTasksColl == 0 && planner->nTasksP2p != 0) {
      //   NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &budget), result, failure);
      // }
      finishPlan(comm, plan);
      if (plan->workBytes != 0) {
        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        nPlans += 1;
      }
    } while (planner->nTasksColl != 0);

    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
    planner->unlaunchedPlansHead = planHead;

    if (nPlans == 0) return ncclSuccess;

    // Semantically we want these dependencies for the kernels launched:
    //   1. Launch host task on hostStream.
    //   2. Launch kernel, depends on all of {deviceStream, hostStream, userStream[i]...}
    //   3. {deviceStream, userStream[i]...} depend on kernel.
    // We achieve this by:
    //   1. userStream[0] waits on deviceStream
    //   2. deviceStream waits on each of userStream[1...]
    //   3. host task launch on hostStream
    //   4. userStream[0] waits on hostStream
    //   5. kernel launch on userStream[0]
    //   6. deviceStream waits on userStream[0]
    //   7. userStream[1...] each waits on deviceStream
    // The two-level fan-in fan-out is because ncclStrongStreamWaitStream() requires
    // at least one of the two streams to be strong-stream.
    cudaStream_t launchStream = planner->streams->stream;
    NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream), result, failure);

    // Create dependency for device stream on user streams. First from extra user
    // streams to deviceStream. Then deviceStream to first user stream.
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, &comm->sharedRes->deviceStream, l->stream), result, failure);
    }
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, launchStream, &comm->sharedRes->deviceStream), result, failure);

    if (persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      bool acquired = false;
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream), result, failure);
          }
          // NCCLCHECKGOTO(ncclStrongStreamLaunchHost(planner->capturingGraph, &comm->sharedRes->hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, launchStream, &comm->sharedRes->hostStream), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream), result, failure);
      }
    }

  }
failure:
  return result;
}

ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call CUDA unless the
  // kernel launch is captured.
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

#if CUDART_VERSION >= 12000
// NCCL uses the "Remote" Mem Sync domain by default
NCCL_PARAM(MemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);
#endif

ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  int nChannels = countOneBits(plan->channelMask);
  void* sym = plan->kernelFn;
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = ncclShmemDynamicSize(comm->cudaArch);
  cudaStream_t launchStream = planner->streams->stream;
  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };

  CUfunction fn;
  CUDACHECK(cudaGetFuncBySymbol(&fn, sym));

  #if CUDART_VERSION >= 11080
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion >= 11080) {
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap == 90) ? comm->config.cgaClusterSize : 0;

    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[3];
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    launchConfig.sharedMemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = launchStream;

    //CUDACHECK(cudaLaunchKernelExC(&launchConfig, fnAddr, args));
    CUCHECK(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra));
    return ncclSuccess;
  }
  #endif
  // Standard kernel launch
  CUCHECK(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra));
  //CUDACHECK(cudaLaunchKernel(fnAddr, grid, block, args, smem, launchStream));
  return ncclSuccess;
}

ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!(plan->persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking)) {
    // We are not using the host stream for proxy ops and reclaimation submission.
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  } else {
    // We are using the host stream for proxy ops and reclaimation submission.
    // Only plans with proxy ops have a callback pushed by ncclLaunchPrepare.
    // Since non-persistent plans also require reclaimation, we have to do it
    // here.
    if (!plan->persistent && !plan->hasProxyOps) {
      ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;

  if (!ncclIntruQueueEmpty(&planner->planQueue)) {
    // Reset queue to empty without destroying plans since those will be sent
    // back to us for reclaiming via callbackQueue.
    ncclIntruQueueConstruct(&planner->planQueue);
    cudaStream_t launchStream = planner->streams->stream; // First user stream gets launch
    // Create dependency for deviceStream on launchStream. We know that deviceStream
    // hasn't been modified since launchStream waited on it (in ncclLaunchPrepare),
    // so we can say that launchStream subsumes it.
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, &comm->sharedRes->deviceStream, launchStream, /*b_subsumes_a=*/true), result, resume1);
  resume1:
    // Create dependency for other user streams (skip launch stream) on deviceStream.
    // Again, the user streams haven't been touched since deviceStream waited on them
    // so we can say they are subsumed by deviceStream.
    struct ncclCudaStreamList* sl = planner->streams->next;
    planner->streams = nullptr; // Reset comm->planner.streams to empty.
    while (sl != nullptr) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, sl->stream, &comm->sharedRes->deviceStream, /*b_subsumes_a=*/true), result, resume2);
    resume2:
      sl = sl->next;
    }
    // Release device stream as acquired in ncclLaunchPrepare()
    NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream), result, resume3);
  resume3:;
  }
  return result;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

static inline ncclResult_t getCollNetSupport(
    struct ncclComm* comm, struct ncclTaskColl* info, int* collNetSupport
  ) {
  // Translate ncclAvg and PreMulSum
  ncclRedOp_t netOp = info->opHost;
  *collNetSupport = comm->collNetSupport;
  switch (info->func) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    *collNetSupport &= comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

static void initCollCostTable(float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;
    }
  }
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t updateCollCostTable(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    int collNetSupport, int nvlsSupport, int numPipeOps,
    float** collCostTable, int* backupAlgo, int* backupProto, float* backupTime
  ) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  if (comm->nRanks == 1) {
    table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
    return ncclSuccess;
  }

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    // CollNetDirect is only supported for up to 8 local GPUs
    if (a == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1 && info->func != ncclFuncAllGather) continue;
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1) continue;
    /* now we only support single-node NVLS allgather and reducescatter */
    if (a == NCCL_ALGO_NVLS && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) && comm->nNodes > 1) continue;
    /* Tree reduceScatter doesn't support scaling yet */
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      bool backup;
      float time;
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &time, &backup));
      if (!backup) {
        table[a][p] = time;
      } else {
        if (time >= 0.0 && time < *backupTime) {
          *backupAlgo = a;
          *backupProto = p;
          *backupTime = time;
        }
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t topoGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    float** collCostTable, int backupAlgo, int backupProto, float backupTime, ncclSimInfo_t* simInfo
  ) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  float minTime = 3600000000.0;
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  int protocol = info->protocol = NCCL_PROTO_UNDEF;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
      if (table[a][p] >= 0.0 && table[a][p] < minTime) {
        algorithm = a;
        protocol = p;
        minTime = table[a][p];
      }
    }
  }

  info->algorithm = algorithm;
  info->protocol = protocol;
  float time = minTime;

  // if NCCL_ALGO environment variable is set as char, use it to override the algorithm
  // and algo, NCCL_PROTO environment variable is set, use it to override the protocol
  char* algoEnv = getenv("NCCL_ALGO");
  // char* protoEnv = getenv("NCCL_PROTO");
  if (algoEnv) {
    if (strcmp(algoEnv, "CollnetDirect") == 0) {
      info->algorithm = NCCL_ALGO_COLLNET_DIRECT;
    }
  }
  info->protocol = NCCL_PROTO_SIMPLE;
  // if (protoEnv) {
  //   int proto = atoi(protoEnv);
  //   if (proto >= 0 && proto < NCCL_NUM_PROTOCOLS) {
  //     info->protocol = proto;
  //   }
  // }

  // Yes, we are first assigning and then testing if protocol is sane, but that's OK in this case.
  // coverity[check_after_sink]
  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    if (backupAlgo == NCCL_ALGO_UNDEF || backupProto == NCCL_PROTO_UNDEF) {
      WARN("Error : no algorithm/protocol available");
      return ncclInternalError;
    }
    info->algorithm = backupAlgo;
    info->protocol = backupProto;
    time = backupTime;
  }
  if (comm->rank == 0) INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %d proto %d time %f", ncclFuncToString(info->func), nBytes, info->algorithm, info->protocol, time);
  if (simInfo) simInfo->estimatedTime = time;
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", nBytes, info->algorithm, info->protocol, time);

  int nc = comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = nBytes < nc*nt*comm->channels[0].collnetDirect.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  // } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
  //   // NVLS should not need more than 16 channels to get peak BW.
  //   nc = comm->nvlsChannels;
  } else {
    // Ring/Tree channel tuning
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  if (info->algorithm != NCCL_ALGO_NVLS && info->algorithm != NCCL_ALGO_NVLS_TREE &&
    info->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
  }
  nt = nt/WARP_SIZE < 3 ? 3*WARP_SIZE : nt;
  if (info->algorithm == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree now uses all threads always.
  if (info->algorithm == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;
  info->nMaxChannels = nc;
  info->nWarps = nt/WARP_SIZE;
  return ncclSuccess;
}

// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, topoGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
static ncclResult_t getAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo/* = NULL*/
  ) {
  size_t nBytes = ncclTypeSize(info->datatype)*ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  int nMaxChannels = 0;
  int backupAlgo = NCCL_ALGO_UNDEF;
  int backupProto = NCCL_PROTO_UNDEF;
  float backupTime = 3600000000.0;
  float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float **)collCostTable);
  NCCLCHECK(updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable, &backupAlgo, &backupProto, &backupTime));
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->getCollInfo(
          comm->tunerContext, info->func, nBytes,
          numPipeOps, (float **)collCostTable, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
          &nMaxChannels));
  }
  NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, backupAlgo, backupProto, backupTime, simInfo));
  info->nMaxChannels = nMaxChannels == 0 ? info->nMaxChannels : nMaxChannels;
  return ncclSuccess;
}

NCCL_PARAM(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

static ncclResult_t calcCollChunking(
    struct ncclComm* comm, struct ncclTaskColl* info, int nChannels, size_t nBytes,
    /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
  ) {
  ncclPattern_t pattern;
  size_t grainSize = ncclProtoGrainSize(info->protocol);

  switch (info->func) {
  case ncclFuncReduceScatter:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatUp :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  case ncclFuncAllGather:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  // case ncclFuncAllReduce:
  //   pattern =
  //     info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
  //     info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
  //     info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
  //     info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
  //     info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
  //     ncclPatternRingTwice;
  //   break;
  default:
    WARN("Unknown pattern for collective %d algorithm %d", info->func, info->algorithm);
    return ncclInternalError;
  }

  int nstepsPerLoop, nchunksPerLoop;
  switch (pattern) {
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternCollnetChain:
    nstepsPerLoop = nchunksPerLoop = 1;
    break;
  case ncclPatternCollnetDirect:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].collnetDirect.nHeads;
    break;
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  int stepSize   = comm->buffSizes[info->protocol];
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize*chunkSteps;
  if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;

  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Optimize chunkSize / nSteps
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 32768) chunkSize /= 2;
  }

  // Compute directFlags of work struct.
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Set direct direction for broadcast-gather (read or write)
    *outDirectFlags = (nBytes/nChannels <= 1024 * 4) ? NCCL_DIRECT_READ : NCCL_DIRECT_WRITE;
  } else {
    *outDirectFlags = 0;
  }

  // Compute nSteps for proxies
  //if (comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->func, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  chunkSize = chunkSize / grainSize * grainSize; // align chunkSize to multiple grainSize
  int nLoops = (int)DIVUP(nBytes, size_t(nChannels)*nchunksPerLoop*chunkSize);
  memset(proxyOp, 0, sizeof(*proxyOp));
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->redOp = info->opHost;
  proxyOp->pattern = pattern;
  proxyOp->coll = info->func;
  proxyOp->root = info->root;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = stepSize*sliceSteps;

  if (info->regBufType == NCCL_COLLNET_REG_BUFFER) {
    proxyOp->reg = 1;
    proxyOp->nsteps = DIVUP(nBytes, NCCL_MAX_COLLNET_SIZE);
    proxyOp->sendMhandle = info->sendMhandle;
    proxyOp->recvMhandle = info->recvMhandle;
    proxyOp->sendbuff = (uint8_t*)info->sendbuff;
    proxyOp->recvbuff = (uint8_t*)info->recvbuff;
    proxyOp->nbytes = nBytes;
  } else {
    proxyOp->reg = 0;
  }

  if (pattern == ncclPatternCollnetDirect) {
    proxyOp->specifics.collnetDirect.nNodes = comm->nNodes;
    proxyOp->specifics.collnetDirect.node = comm->node;
    if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
      proxyOp->specifics.collnetDirect.sizePerRank = info->count*ncclTypeSize(info->datatype);
    }
  }

  if (pattern == ncclPatternPatUp || pattern == ncclPatternPatDown) {
    proxyOp->nbytes = DIVUP(nBytes, nChannels);
  }

  *outChunkSize = chunkSize;
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t   i8; uint8_t   u8;
    int32_t i32; uint32_t u32;
    int64_t i64; uint64_t u64;
    half f16; float f32; double f64;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8*ncclTypeSize(datatype);
  if (nbits <= 0) return ncclInvalidArgument;

  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

// Converts `info` to a task and adds it to `comm->planner`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
  //   int peer = info->root;
  //   ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
  //   bool isSendNotRecv = info->coll == ncclFuncSend;

  //   // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  //   ncclGroupCommJoin(info->comm);
  //   struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);
  //   p2p->func = info->coll;
  //   p2p->buff = (void*)info->recvbuff;
  //   p2p->count = info->count;
  //   p2p->datatype = info->datatype;
  //   p2p->root = info->root;
  //   p2p->bytes = nBytes;
  //   ncclIntruQueueEnqueue(
  //     isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue,
  //     p2p);
  //   planner->nTasksP2p += 1;

  //   // Mark channels that need pre-connect
  //   if (comm->rank != peer) {
  //     if (!(isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen)) {
  //       (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true;
  //       int round = 0;
  //       while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
  //                                     : comm->p2pSchedule[round].recvRank)) {
  //         round += 1;
  //       }
  //       uint8_t base = ncclP2pChannelBaseForRound(comm, round);
  //     }
  //   }
  // } else {
  {
    // Empty collectives can be discarded.
    if (info->count == 0) return ncclSuccess;

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    struct ncclDevRedOpFull opDev;
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

    // if (comm->nRanks == 1) {
    //   NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
    //   return ncclSuccess;
    // } else
    {
      // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
      ncclGroupCommJoin(info->comm);
      struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      t->func = info->coll;
      t->sendbuff = info->sendbuff;
      t->recvbuff = info->recvbuff;
      t->count = info->count;
      t->root = info->root;
      t->datatype = info->datatype;
      size_t elementSize = ncclTypeSize(t->datatype);
      if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
        t->count *= elementSize;
        t->datatype = ncclInt8;
        elementSize = 1;
      }
      t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
      t->opHost = info->op;
      t->opDev = opDev; // C++ struct assignment
      t->chunkSteps = info->chunkSteps;
      t->sliceSteps = info->sliceSteps;

      planner->nTasksColl += 1;
      ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
    }
  }

  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  NCCLCHECK(ncclGroupStartInternal());
  ncclResult_t ret = ncclSuccess;
  int devOld = -1;

  NCCLCHECKGOTO(CommCheck(info->comm, info->opName, "comm"), ret, fail);
  // Check whether communicator is ready to communicate
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  if (info->comm->checkPointers) {
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret)); }
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}
