/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  // case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  default: return "Invalid";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}

// NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
//     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
// ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
//     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
//   struct NvtxParamsAllReduce {
//     size_t bytes;
//     ncclRedOp_t op;
//   };
//   // Just pass the size of one message and not the total bytes sent/received.
//   static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
//     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
//     {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
//       offsetof(NvtxParamsAllReduce, op)}
//   };
//   NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
//   NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

//   struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
//     sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
//     ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
//   NCCLCHECK(ncclEnqueueCheck(&info));
//   return ncclSuccess;
// }

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};
