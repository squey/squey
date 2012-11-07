
#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/common.h>

#include "PVTestQuadTree.h"

#include <tbb/task_scheduler_init.h>

#define ZZT_BBITS (PARALLELVIEW_ZZT_BBITS)

typedef Test::PVQuadTree<10000, 1000, 0, ZZT_BBITS> quadtree_t;
typedef Test::PVQuadTreeEntry                       quadtree_entry_t;
typedef quadtree_t::pv_tlr_buffer_t                 tlr_buffer_t;
typedef tlr_buffer_t::index_t                       tlr_index_t;
typedef quadtree_t::insert_entry_f                  insert_entry_f;
typedef quadtree_t::visit_context_t                 visit_context_t;
typedef PVParallelView::PVBCICode<ZZT_BBITS>        bci_code_t;
typedef PVParallelView::constants<ZZT_BBITS>        constants;

constexpr static uint32_t mask_int_ycoord = constants::mask_int_ycoord;

/*****************************************************************************
 * about data
 */
#define QT_MAX_VALUE (1U << 22)
#define QT_MASK (QT_MAX_VALUE -1)

size_t init_count(size_t num)
{
	return num * num;
}

quadtree_entry_t *init_entries(size_t num)
{
	typedef PVCore::PVAlignedAllocator<quadtree_entry_t, 16> Alloc;
	quadtree_entry_t *entries = Alloc().allocate(num*num);
	size_t idx = 0;
	for(size_t i = 0; i < num; ++i) {
		for(size_t j = 0; j < num; ++j) {
			entries[i].idx = idx;
			entries[i].y1 = rand() & QT_MASK;
			entries[i].y2 = rand() & QT_MASK;
			++idx;
		}
	}
	return entries;
}


void print_time(const char *text, double s)
{
	std::cout << text << ": " << s * 1000. << " ms" << std::endl;
}

/*****************************************************************************
 * copy/paste from namespace PVParallelView
 *****************************************************************************/

static inline double get_scale_factor(uint32_t zoom)
{
	return pow(2, zoom);
}

static inline void compute_bci_projection_y2(const uint64_t y1,
                                             const uint64_t y2,
                                             const uint64_t y_min,
                                             const uint64_t y_lim,
                                             const int shift,
                                             const uint32_t mask,
                                             const uint32_t width,
                                             const float beta,
                                             bci_code_t &bci)
{
	bci.s.l = ((y1 - y_min) >> shift) & mask;

	int64_t dy = (int64_t)y2 - (int64_t)y1;
	double y2p = (double)y1 + dy * (double)beta;

	if (y2p >= y_lim) {
		bci.s.type = bci_code_t::DOWN;
		bci.s.r = ((double)width * (double)(y_lim - y1)) / (double)(y2p - y1);
	} else if (y2p < y_min) {
		bci.s.type = bci_code_t::UP;
		bci.s.r = ((double)width * (double)(y1 - y_min)) / (double)(y1 - y2p);
	} else {
		bci.s.type = bci_code_t::STRAIGHT;
		bci.s.r = (((uint32_t)(y2p - y_min)) >> shift) & mask;
	}
}


/*****************************************************************************
 * about program's parameters
 */
enum {
	P_PROG = 0,
	P_NUM,
	P_ZOOM,
	P_MIN,
	P_MAX_VALUE
};

void usage(const char *program)
{
	std::cerr << "usage: " << basename(program) << " num zoom min\n" << std::endl;
	std::cerr << "\tnum  : number of values used for each coordinate" << std::endl;
	std::cerr << "\tzoom : zoom level in [0,21]" << std::endl;
	std::cerr << "\tmin  : low value of extraction range" << std::endl;
}

/*****************************************************************************
 * main
 */

#define QT_MIN 0
#define QT_MAX (QT_MAX_VALUE)
#define QT_HEIGHT 8

int main(int argc, char **argv)
{
	if (argc != P_MAX_VALUE) {
		usage(argv[P_PROG]);
		exit(1);
	}

	tbb::task_scheduler_init init(8);

	size_t num = (size_t) atol(argv[P_NUM]);
	uint32_t zoom = (uint32_t) atol(argv[P_ZOOM]);

	const uint64_t y1_min = (uint64_t) atol(argv[P_MIN]);
	const uint64_t y1_max = y1_min + (1UL << (32 - zoom));
	const uint64_t y1_lim = y1_max;
	std::cout << "y1_min  : " << y1_min << std::endl;
	std::cout << "y1_max  : " << y1_max << std::endl;
	std::cout << "y1_lim  : " << y1_lim << std::endl;

	quadtree_t *qt = new quadtree_t(QT_MIN, QT_MAX, QT_MIN, QT_MAX, QT_HEIGHT);

	BENCH_START(data);
	quadtree_entry_t *entries = init_entries(num);
	for(size_t i = 0; i < init_count(num); ++i) {
		qt->insert(entries[i]);
	}
	BENCH_STOP(data);
	print_time("data", BENCH_END_TIME(data));

	const uint64_t y_min = y1_min;
	const uint64_t y_lim = y1_lim;
	const uint32_t shift = (32 - PARALLELVIEW_ZZT_BBITS) - zoom;
	const uint32_t width = 512;
	double beta = 1. / get_scale_factor(zoom);

	const insert_entry_f insert_f([&](const quadtree_entry_t &e, tlr_buffer_t &buffer)
	                              {
		                              bci_code_t bci;
		                              compute_bci_projection_y2(e.y1, e.y2,
		                                                        y_min, y_lim,
		                                                        shift, mask_int_ycoord,
		                                                        width, beta, bci);
		                              tlr_index_t tlr(bci.s.type,
		                                              bci.s.l,
		                                              bci.s.r);
		                              if (e.idx < buffer[tlr.v]) {
			                              buffer[tlr.v] = e.idx;
		                              }
	                              });

	tlr_buffer_t *tlr_seq = new tlr_buffer_t;
	uint32_t *buffer_seq = new uint32_t [2048 * 4096];

	BENCH_START(seq);
	qt->get_first_from_y1(y1_min, y1_max, zoom, 4096,
	                      buffer_seq, insert_f, *tlr_seq);
	BENCH_STOP(seq);
	print_time("seq", BENCH_END_TIME(seq));

	visit_context_t *ctx = new visit_context_t(y1_min, y1_max, insert_f);

	BENCH_START(tbb);
	/* extraction
	 */
	std::cout << "tbb extraction" << std::endl;
	qt->get_first_from_y1(*ctx, zoom, 4096);
	BENCH_STOP(tbb);
	print_time("tbb", BENCH_END_TIME(tbb));

	/* merge of TLR buffers
	 */
	std::cout << "tls merge" << std::endl;
	visit_context_t::tls_set_t::iterator it = ctx->get_tls().begin();
	tlr_buffer_t &tlr_tbb = it->get_tlr_buffer();

	++it;
	for(;it != ctx->get_tls().end(); ++it) {
		tlr_buffer_t &tlr_buffer2 = it->get_tlr_buffer();

		for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
			if(tlr_buffer2[i] < tlr_tbb[i]) {
				tlr_tbb[i] = tlr_buffer2[i];
			}
		}

		tlr_buffer2.clear();
	}

	int seq_found_count = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_seq)[i] != PVROW_INVALID_VALUE) {
			++seq_found_count;
		}
	}
	std::cout << "BCI codes found in seq version: " << seq_found_count << std::endl;

	int tbb_found_count = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((tlr_tbb)[i] != PVROW_INVALID_VALUE) {
			++tbb_found_count;
		}
	}
	std::cout << "BCI codes found in tbb version: " << tbb_found_count << std::endl;

	bool diff = false;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_seq)[i] != (tlr_tbb)[i]) {
			diff = true;
			break;
		}
	}

	if (diff) {
		std::cout << "TLR buffers are different" << std::endl;
	} else {
		std::cout << "TLR buffers are equal" << std::endl;
		std::cout << "speed-up: " << BENCH_END_TIME(seq) / BENCH_END_TIME(tbb)
		          << std::endl;
	}

	return 0;
}
