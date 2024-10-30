/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    int64_t value = deftVal;
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&mutex);
}

const char* ncclGetEnv(const char* name) {
  return getenv(name);
}
