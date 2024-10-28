/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"

#define MAXNAMESIZE 64
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

static int ncclNMergedIbDevs = -1;
#define NCCL_IB_MAX_DEVS_PER_NIC 2
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  int ndevs;
  int devs[NCCL_IB_MAX_DEVS_PER_NIC]; // Points to an index in ncclIbDevs
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
  int dmaBufSupported;               //  0 = uninit, 1 = yes, -1 = no
};

struct ncclIbStats {
  int fatalErrorCount;
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  int realPort;
  int maxQp;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
};

#define MAX_IB_DEVS 32
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_DEVS];
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclIbRelaxedOrderingEnabled = 0;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", 0);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    WARN("communicator encountered a fatal error (detected in %s)\n", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}

pthread_t ncclIbAsyncThread;
static void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got async error event: %s", dev->devName, dev->portNum, str);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000 /* NDR */ };

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

// Compare ncclIbDev[dev] to all stored mergedIbDevs
int ncclIbFindMatchingDev(int dev) {
  for (int i = 0; i < ncclNMergedIbDevs; i++) {
    if (ncclIbMergedDevs[i].ndevs < NCCL_IB_MAX_DEVS_PER_NIC) {
      int compareDev = ncclIbMergedDevs[i].devs[0];
      if (strcmp(ncclIbDevs[dev].pciPath, ncclIbDevs[compareDev].pciPath) == 0 &&
          (ncclIbDevs[dev].guid == ncclIbDevs[compareDev].guid) &&
          (ncclIbDevs[dev].link == ncclIbDevs[compareDev].link)) {
          TRACE(NCCL_NET, "NET/IB: Matched name1=%s pciPath1=%s guid1=0x%lx link1=%u name2=%s pciPath2=%s guid2=0x%lx link2=%u",
            ncclIbDevs[dev].devName, ncclIbDevs[dev].pciPath, ncclIbDevs[dev].guid, ncclIbDevs[dev].link,
            ncclIbDevs[compareDev].devName, ncclIbDevs[compareDev].pciPath, ncclIbDevs[compareDev].guid, ncclIbDevs[compareDev].link);
          return i;
      }
    }
  }

  return ncclNMergedIbDevs;
}

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  ncclResult_t ret = ncclSuccess;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      if (ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      // Should NCCL merge multi-port devices into one?
      int mergeNics;
      mergeNics = ncclParamIbMergeNics();
build_ib_list:
      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
          ncclIbDevs[ncclNIbDevs].portNum = port_num;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].pdRefs = 0;
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
          NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

          TRACE(NCCL_NET,"NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, ncclIbDevs[ncclNIbDevs].portNum,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE", ncclIbDevs[ncclNIbDevs].speed, context, ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

          PTHREADCHECKGOTO(pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs), "pthread_create", ret, fail);
          ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
          PTHREADCHECKGOTO(pthread_detach(ncclIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d

          int mergedDev = ncclNMergedIbDevs;
          if (mergeNics) {
            mergedDev = ncclIbFindMatchingDev(ncclNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if there's only one dev inside)
          if (mergedDev == ncclNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IB device
            ncclIbMergedDevs[mergedDev].ndevs = 1;
            ncclIbMergedDevs[mergedDev].devs[0] = ncclNIbDevs;
            ncclNMergedIbDevs++;
            strncpy(ncclIbMergedDevs[mergedDev].devName, ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE);
          // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IB device
            int ndevs = ncclIbMergedDevs[mergedDev].ndevs;
            ncclIbMergedDevs[mergedDev].devs[ndevs] = ncclNIbDevs;
            ncclIbMergedDevs[mergedDev].ndevs++;
            snprintf(ncclIbMergedDevs[mergedDev].devName + strlen(ncclIbMergedDevs[mergedDev].devName), MAXNAMESIZE+1, "+%s", ncclIbDevs[ncclNIbDevs].devName);
          }

          // Aggregate speed
          ncclIbMergedDevs[mergedDev].speed += ncclIbDevs[ncclNIbDevs].speed;
          ncclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      // Detect if there are both multi-port and single-port NICs in the system. If so, disable port merging and build the list again
      if (mergeNics) {
        for (int d = 0; d < ncclNMergedIbDevs; d++) {
          if (ncclIbMergedDevs[d].ndevs != ncclIbMergedDevs[0].ndevs) {
            INFO(NCCL_NET, "Detected a mix of single and multiple-port NICs. Force-disabling NCCL_IB_MERGE_NICS");
            mergeNics = 0;
            ncclNIbDevs = 0;
            ncclNMergedIbDevs = 0;
            memset(ncclIbMergedDevs, 0, sizeof(ncclIbMergedDevs));
            goto build_ib_list;
          }
        }
      }

      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      for (int d = 0; d < ncclNMergedIbDevs; d++) {
        struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibDev = mergedDev->devs[i];
            snprintf(line+strlen(line), 2047-strlen(line), "[%d] %s:%d/%s%s", ibDev, ncclIbDevs[ibDev].devName,
              ncclIbDevs[ibDev].portNum, ncclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              // Insert comma to delineate
              i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line+strlen(line), 2047-strlen(line), "}");
        } else {
          int ibDev = mergedDev->devs[0];
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]%s:%d/%s", ibDev, ncclIbDevs[ibDev].devName,
            ncclIbDevs[ibDev].portNum, ncclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
           ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&ncclIbLock);
  goto exit;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int ncclIbGdrModuleLoaded = 0; // 1 = true, 0 = false
static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}
ncclResult_t ncclIbGdrSupport() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, ibGdrSupportInitOnce);
  if (!ncclIbGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

static __thread int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce(){
  ncclResult_t res;
  // select the appropriate
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  // Test each real devices
  int dev_fail = 0;
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDev = mergedDev->devs[i];
    struct ibv_pd* pd;
    struct ibv_context* ctx = ncclIbDevs[ibDev].context;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
    // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
    dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
    // stop the search and goto failure
    if (dev_fail) goto failure;
  }
  mergedDev->dmaBufSupported = 1;
  return;
failure:
  mergedDev->dmaBufSupported = -1;
  return;
}
// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  struct oncewrap {
    pthread_once_t once = PTHREAD_ONCE_INIT;
  };
  static oncewrap onces[MAX_IB_DEVS];
  // init the device only once
  ibDmaSupportInitDev = dev;
  pthread_once(&onces[dev].once, ibDmaBufSupportInitOnce);

  int dmaBufSupported = ncclIbMergedDevs[dev].dmaBufSupported;
  if (dmaBufSupported == 1) return ncclSuccess;
  return ncclSystemError;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs+dev;
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  struct ncclIbDev* ibDev = ncclIbDevs + mergedDev->devs[0];
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  NULL /* ncclIbListen */,
  NULL /* ncclIbConnect */,
  NULL /* ncclIbAccept */,
  NULL /* ncclIbRegMr */,
  NULL /* ncclIbRegMrDmaBuf */,
  NULL /* ncclIbDeregMr */,
  NULL /* ncclIbIsend */,
  NULL /* ncclIbIrecv */,
  NULL /* ncclIbIflush */,
  NULL /* ncclIbTest */,
  NULL /* ncclIbCloseSend */,
  NULL /* ncclIbCloseRecv */,
  NULL /* ncclIbCloseListen */,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

