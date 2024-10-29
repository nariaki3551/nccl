/*************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GDRWRAP_H_
#define NCCL_GDRWRAP_H_

#include "nccl.h"
#include "alloc.h"
#include <stdint.h> // for standard [u]intX_t types
#include <stdio.h>
#include <stdlib.h>

// These can be used if the GDR library isn't thread safe
#include <pthread.h>
extern pthread_mutex_t gdrLock;
#define GDRLOCK() pthread_mutex_lock(&gdrLock)
#define GDRUNLOCK() pthread_mutex_unlock(&gdrLock)
#define GDRLOCKCALL(cmd, ret) do {                      \
    GDRLOCK();                                          \
    ret = cmd;                                          \
    GDRUNLOCK();                                        \
} while(false)

#define GDRCHECK(cmd) do {                              \
    int e;                                              \
    /* GDRLOCKCALL(cmd, e); */                          \
    e = cmd;                                            \
    if( e != 0 ) {                                      \
      WARN("GDRCOPY failure %d", e);                    \
      return ncclSystemError;                           \
    }                                                   \
} while(false)

// This is required as the GDR memory is mapped WC
#if !defined(__NVCC__)
#if defined(__PPC__)
static inline void wc_store_fence(void) { asm volatile("sync") ; }
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wc_store_fence(void) { std::atomic_thread_fence(std::memory_order_release); }
#else
#include <stdatomic.h>
static inline void wc_store_fence(void) { atomic_thread_fence(memory_order_release); }
#endif
#endif
#endif

// //#define GDR_DIRECT 1
// // Dynamically handle dependency the GDR API library

// /* Extracted from gdrapi.h (v2.1 Nov 2020) */

// #define GPU_PAGE_SHIFT   16
// #define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
// #define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
// #define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

// struct gdr;
// typedef struct gdr *gdr_t;

// /* End of gdrapi.h */

// // Global GDR driver handle
// extern gdr_t ncclGdrCopy;

// #include "alloc.h"

#endif // End include guard
