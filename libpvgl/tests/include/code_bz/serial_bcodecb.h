#ifndef SERIAL_BCODECB_H
#define SERIAL_BCODECB_H

#include <common/common.h>
#include <code_bz/types.h>
#include <code_bz/bcode_cb.h>
#include <cstdlib>

void serial_bcodecb(PVBCode* pts, size_t n, BCodeCB cb);
void sse_bcodecb(PVBCode* pts, size_t n, BCodeCB cb);

#endif
