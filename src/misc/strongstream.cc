/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "strongstream.h"
#include "cudawrap.h"
#include "checks.h"
#include "param.h"

// Tracks the chain of graph nodes for a given graph captured identified by
// its graph id. This state has to live for as long as captured work is being
// submitted. CUDA doesn't have mechanism to inform us when the user ends capture
// so the best we can do is get notified when the graph is destroyed.
struct ncclStrongStreamGraph {
  struct ncclStrongStreamGraph* next;
  // Atomically exchanged to false by both the main thread or the graph destructor
  // callback. The last to arrive deletes the node.
  bool alive;
  unsigned long long graphId;
  // For each graph we track the "tip" of the chain of graph nodes. A linear
  // chain would always have just one node at its tip, but since we have to merge
  // in chains from other streams (via ncclStrongStreamWaitStream) some spots
  // in the chain can be wider than a single node and thus need a list, so we
  // maintain a dynamically sized array of tip nodes.
  int tipCount, tipCapacity;
  cudaGraphNode_t* tipNodes;
};

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream
  ) {
  #if CUDART_VERSION >= 10000 // cudaStreamGetCaptureInfo
    int driver;
    NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (CUDART_VERSION < 11030 || driver < 11030) {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      CUDACHECK(cudaStreamGetCaptureInfo(stream, &status, &gid));
      #if CUDART_VERSION >= 11030
        graph->graph = nullptr;
        graph->graphId = ULLONG_MAX;
      #endif
      if (status != cudaStreamCaptureStatusNone) {
        WARN("NCCL cannot be captured in a graph if either it wasn't built with CUDA runtime >= 11.3 or if the installed CUDA driver < R465.");
        return ncclInvalidUsage;
      }
    } else {
      #if CUDART_VERSION >= 11030
        cudaStreamCaptureStatus status;
        unsigned long long gid;
        CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &gid, &graph->graph, nullptr, nullptr));
        if (status != cudaStreamCaptureStatusActive) {
          graph->graph = nullptr;
          gid = ULLONG_MAX;
        }
        graph->graphId = gid;
      #endif
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg) {
  #if CUDART_VERSION >= 11030
    cudaUserObject_t object;
    CUDACHECK(cudaUserObjectCreate(
      &object, arg, fn, /*initialRefcount=*/1, cudaUserObjectNoDestructorSync
    ));
    // Hand over ownership to CUDA Graph
    CUDACHECK(cudaGraphRetainUserObject(graph.graph, object, 1, cudaGraphUserObjectMove));
    return ncclSuccess;
  #else
    return ncclInvalidUsage;
  #endif
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaStreamCreateWithFlags(&ss->cudaStream, cudaStreamNonBlocking));
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaEventCreateWithFlags(&ss->serialEvent, cudaEventDisableTiming));
    ss->everCaptured = false;
    ss->serialEventNeedsRecord = false;
    ss->graphHead = nullptr;
  #else
    CUDACHECK(cudaEventCreateWithFlags(&ss->scratchEvent, cudaEventDisableTiming));
  #endif
  return ncclSuccess;
}

NCCL_PARAM(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", 1)

static void ensureTips(struct ncclStrongStreamGraph* g, int n) {
  if (g->tipCapacity < n) {
    g->tipNodes = (cudaGraphNode_t*)realloc(g->tipNodes, n*sizeof(cudaGraphNode_t));
    g->tipCapacity = n;
  }
}

ncclResult_t ncclStrongStreamAcquire(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss
  ) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (graph.graph == nullptr) {
      if (mixing && ss->everCaptured) {
        CUDACHECK(cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
        ss->serialEventNeedsRecord = false;
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (mixing && ss->everCaptured) {
      CUDACHECK(cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
    }
    ss->serialEventNeedsRecord = true; // Assume the caller is going to add work to stream.
  #endif
  return ncclSuccess;
}

static ncclResult_t checkGraphId(struct ncclStrongStreamGraph* g, unsigned long long id) {
  if (g == nullptr || g->graphId != id) {
    WARN("Expected graph id=%llu was not at head of strong stream's internal list.", id);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (mixing && ss->serialEventNeedsRecord) {
      if (graph.graph == nullptr) {
        if (ss->everCaptured) {
          CUDACHECK(cudaEventRecord(ss->serialEvent, ss->cudaStream));
          ss->serialEventNeedsRecord = false;
        }
      } else {
        struct ncclStrongStreamGraph* g = ss->graphHead;
        NCCLCHECK(checkGraphId(g, graph.graphId));
        ensureTips(g, 1);
        CUDACHECK(cudaGraphAddEventRecordNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, ss->serialEvent));
        g->tipCount = 1;
        ss->serialEventNeedsRecord = false;
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchKernel(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss,
    void* fn, dim3 grid, dim3 block, void* args[], size_t sharedMemBytes
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->cudaStream));
    } else {
      cudaKernelNodeParams p;
      p.func = fn;
      p.gridDim = grid;
      p.blockDim = block;
      p.kernelParams = args;
      p.sharedMemBytes = sharedMemBytes;
      p.extra = nullptr;
      struct ncclStrongStreamGraph* g = ss->graphHead;
      NCCLCHECK(checkGraphId(g, graph.graphId));
      ensureTips(g, 1);
      CUDACHECK(cudaGraphAddKernelNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, &p));
      g->tipCount = 1;
    }
    ss->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->cudaStream));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->serialEventNeedsRecord) {
        b->serialEventNeedsRecord = false;
        CUDACHECK(cudaEventRecord(b->serialEvent, b->cudaStream));
      }
      CUDACHECK(cudaStreamWaitEvent(a->cudaStream, b->serialEvent, 0));
    } else {
      struct ncclStrongStreamGraph* ag = a->graphHead;
      NCCLCHECK(checkGraphId(ag, graph.graphId));
      struct ncclStrongStreamGraph* bg = b->graphHead;
      NCCLCHECK(checkGraphId(bg, graph.graphId));
      if (b_subsumes_a) ag->tipCount = 0;
    }
    a->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaEventRecord(b->scratchEvent, b->cudaStream));
    CUDACHECK(cudaStreamWaitEvent(a->cudaStream, b->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      // It is ok to use a->serialEvent to record b since we'll be setting
      // a->serialEventNeedsRecord so the event won't be considered accurate
      // until re-recorded.
      CUDACHECK(cudaEventRecord(a->serialEvent, b));
      CUDACHECK(cudaStreamWaitEvent(a->cudaStream, a->serialEvent, 0));
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long bGraphId;
      cudaGraphNode_t const* bNodes;
      size_t bCount = 0;
      CUDACHECK(cudaStreamGetCaptureInfo_v2(b, &status, &bGraphId, nullptr, &bNodes, &bCount));
      if (status != cudaStreamCaptureStatusActive || graph.graphId != bGraphId) {
        WARN("Stream is not being captured by the expected graph.");
        return ncclInvalidUsage;
      }
      struct ncclStrongStreamGraph* ag = a->graphHead;
      NCCLCHECK(checkGraphId(ag, graph.graphId));
      if (b_subsumes_a) ag->tipCount = 0;
    }
    a->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaEventRecord(a->scratchEvent, b));
    CUDACHECK(cudaStreamWaitEvent(a->cudaStream, a->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->serialEventNeedsRecord) {
        b->serialEventNeedsRecord = false;
        CUDACHECK(cudaEventRecord(b->serialEvent, b->cudaStream));
      }
      CUDACHECK(cudaStreamWaitEvent(a, b->serialEvent, 0));
    } else {
      struct ncclStrongStreamGraph* bg = b->graphHead;
      NCCLCHECK(checkGraphId(bg, graph.graphId));
      CUDACHECK(cudaStreamUpdateCaptureDependencies(a, bg->tipNodes, bg->tipCount,
        b_subsumes_a ? cudaStreamSetCaptureDependencies : cudaStreamAddCaptureDependencies
      ));
    }
  #else
    CUDACHECK(cudaEventRecord(b->scratchEvent, b->cudaStream));
    CUDACHECK(cudaStreamWaitEvent(a, b->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
    ss->serialEventNeedsRecord = false;
  #endif
  CUDACHECK(cudaStreamSynchronize(ss->cudaStream));
  return ncclSuccess;
}
