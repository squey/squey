#ifndef SERIAL_BCODECB_H
#define SERIAL_BCODECB_H

#include <common/common.h>
#include <code_bz/types.h>
#include <code_bz/bcode_cb.h>
#include <cstdlib>

void omp_bcodecb(PVBCode* pts, size_t n, BCodeCB cb, BCodeCB* cb_threads);
void omp_bcodecb_atomic(PVBCode* pts, size_t n, BCodeCB cb);
void omp_bcodecb_atomic2(PVBCode* pts, size_t n, BCodeCB cb);
void omp_bcodecb_sse_branch(PVBCode* pts, size_t n, BCodeCB cb);
void serial_bcodecb(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_tile(PVBCode* pts, size_t n, BCodeCB cb, uint32_t** tiles_cb);
void bcodecb_branch(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_stream(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_sse(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_sse_branch(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_sse_branch2(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_sort_unique(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_parallel_sort_unique(PVBCode* pts, size_t n, BCodeCB cb);
void bcodecb_sse_sort_branch(PVBCode* codes, size_t n, BCodeCB cb);

#endif
