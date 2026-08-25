//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/squey_intrin.h>
#include <pvhwloc.h>

#include <squey/PVSelection.h>

#include <pvparallelview/PVHitGraphDataOMP.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

//#include <numa.h>
#include <omp.h>

// Constants used by the OMP code
//#define NBITS (PVParallelView::PVHitGraphCommon::NBITS) // Number of bits used
// by the reduction
//#define BUFFER_SIZE (PVParallelView::PVHitGraphBuffer::SIZE_BLOCK) // Number
// of integers in one block

//
// OMP specific context structure
//

// One of the two builds of the hit graph kernels : this unit is compiled with the
// baseline instruction set, its twin with -mavx2, and process_all() picks between
// them at runtime. Keep both in sync by editing the shared .ipp only.
#include <pvkernel/core/squey_intrin.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

namespace hitgraph_baseline
{
#include "PVHitGraphDataOMP_kernels.ipp"
} // namespace hitgraph_baseline
