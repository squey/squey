#ifndef BCODECB_H
#define BCODECB_H

#include <common/common.h>
#include <code_bz/types.h>
#include <stdint.h>
#include <vector>

typedef uint32_t* DECLARE_ALIGN(16) BCodeCB;

BCodeCB allocate_BCodeCB(void);
void free_BCodeCB(BCodeCB b);
void bcode_cb_to_bcodes(std::vector<PVBCode>& ret, BCodeCB cb);


#define NB_BITS_BCODE 25 // 11*2 for 'l' and 'r', and 3 for the type
#define NB_BCODE (1<<(NB_BITS_BCODE))
#define NB_INT_BCODECB (NB_BCODE/(8*sizeof(uint32_t)))
#define SIZE_BCODECB (NB_INT_BCODECB*sizeof(uint32_t))

#endif
