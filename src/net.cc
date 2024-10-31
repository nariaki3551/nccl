/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "bootstrap.h"
#include "checks.h"

#include <string.h>
#include <errno.h>
#include <dlfcn.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
ncclNet_t* ncclNets[3] = { nullptr, &ncclNetIb, nullptr };
ncclCollNet_t* ncclCollNets[3] = { nullptr, nullptr, nullptr };
enum ncclNetState {
  ncclNetStateInit = 0,
  ncclNetStateEnabled = 1,
  ncclNetStateDisabled = 2
};
enum ncclNetState ncclNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };
enum ncclNetState ncclCollNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

#define MAX_STR_LEN 255

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static void* openNetPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char netPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  {
    snprintf(netPluginLibName, PATH_MAX, "libnccl-net.so");
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
  }
  return nullptr;
}

static pthread_mutex_t netPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int netPluginRefCount;
static void* netPluginLib;

enum {
  netPluginLoadFailed  = -1,
  netPluginLoadReady   =  0,
  netPluginLoadSuccess =  1,
};

static int netPluginStatus = netPluginLoadReady;

#define MAX_PLUGIN_LOAD 2

ncclResult_t ncclNetPluginLoad(struct ncclComm* comm) {
  char couldNotFindNames[MAX_PLUGIN_LOAD * PATH_MAX] = { 0 };
  pthread_mutex_lock(&netPluginLock);
  if (netPluginLoadFailed == netPluginStatus) {
    goto exit;
  }
  if (netPluginLoadSuccess == netPluginStatus) {
    ++netPluginRefCount;
    goto exit;
  }

  netPluginLib = openNetPluginLib(couldNotFindNames, MAX_PLUGIN_LOAD * PATH_MAX);
  if (netPluginLib == nullptr) {
    if (strlen(couldNotFindNames)) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Could not find:%s. Using internal network plugin.", couldNotFindNames);
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Using internal network plugin.");
    }
    goto fail;
  }

  ncclNets[0] = (ncclNet_v8_t*)dlsym(netPluginLib, "ncclNetPlugin_v8");

  // Check for CollNet
  ncclCollNets[0] = (ncclCollNet_v8_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v8");
  ncclCollNets[1] = ncclCollNets[0];
  ncclCollNets[2] = ncclCollNets[0];

  ++netPluginRefCount;
  netPluginStatus = netPluginLoadSuccess;
  comm->netPluginLoaded = 1;

exit:
  pthread_mutex_unlock(&netPluginLock);
  return ncclSuccess;
fail:
  if (netPluginLib) dlclose(netPluginLib);
  netPluginStatus = netPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclNetPluginUnload(struct ncclComm* comm) {
  pthread_mutex_lock(&netPluginLock);
  if (comm->netPluginLoaded && 0 == (--netPluginRefCount)) {
    if (ncclNets[0]) {
      INFO(NCCL_NET, "NET/Plugin: Closing net plugin '%s'", ncclNets[0]->name);
    }
    if (ncclCollNets[0]) {
      INFO(NCCL_NET, "NET/Plugin: Closing collnet plugin '%s'", ncclCollNets[0]->name);
    }
    dlclose(netPluginLib);
    netPluginLib = nullptr;
    ncclNets[0] = nullptr;
    ncclCollNets[0] = nullptr;
    netPluginStatus = netPluginLoadReady;
    comm->netPluginLoaded = 0;
  }
  pthread_mutex_unlock(&netPluginLock);
  return ncclSuccess;
}

static ncclResult_t netGetState(int i, enum ncclNetState* state) {
  pthread_mutex_lock(&netLock);
  if (ncclNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclNets[i]->init(ncclDebugLog) != ncclSuccess) ncclNetStates[i] = ncclNetStateDisabled;
    else if (ncclNets[i]->devices == nullptr) ncclNetStates[i] = ncclNetStateDisabled;
    else if (ncclNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclNetStates[i] = ncclNetStateDisabled;
    else ncclNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclNetStates[i];
  pthread_mutex_unlock(&netLock);
  return ncclSuccess;
}

static ncclResult_t collNetGetState(int i, enum ncclNetState* state) {
  pthread_mutex_lock(&netLock);
  if (ncclCollNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclCollNets[i]->init(ncclDebugLog) != ncclSuccess) ncclCollNetStates[i] = ncclNetStateDisabled;
    else if (ncclCollNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclCollNetStates[i] = ncclNetStateDisabled;
    else ncclCollNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclCollNetStates[i];
  pthread_mutex_unlock(&netLock);
  return ncclSuccess;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  // Initialize main communication network
  const char* netName;
  bool ok = false;

  netName = comm->config.netName;
  for (int i=0; i<3; i++) {
    if (ncclNets[i] == nullptr) continue;
    enum ncclNetState state;
    NCCLCHECK(netGetState(i, &state));
    if (state != ncclNetStateEnabled) continue;
    if (netName && strcasecmp(netName, ncclNets[i]->name) != 0) continue;

    comm->ncclNet = ncclNets[i];
    ok = true;

    if (ncclCollNets[i]) {
      NCCLCHECK(collNetGetState(i, &state));
      if (state == ncclNetStateEnabled) {
        comm->ncclCollNet = ncclCollNets[i];
      }
    }
    break;
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}