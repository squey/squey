
#include <iostream>

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVQuadTree.h>

#define QT_MAX_VALUE (BUCKET_ELT_COUNT)

/*****************************************************************************
 * lots of type :-P
 */
typedef PVParallelView::PVQuadTree<10000, 1000, 0, PARALLELVIEW_ZZT_BBITS> quadtree_t;
typedef PVParallelView::PVQuadTreeEntry                                    quadtree_entry_t;
typedef PVParallelView::pv_quadtree_buffer_entry_t                         quadtree_buffer_entry_t;
typedef quadtree_t::insert_entry_f                                         insert_entry_f;
typedef PVParallelView::PVBCICode<PARALLELVIEW_ZZT_BBITS>                  bci_code_t;
typedef quadtree_t::pv_tlr_buffer_t                                        tlr_buffer_t;
typedef tlr_buffer_t::index_t                                              tlr_index_t;

typedef PVParallelView::constants<PARALLELVIEW_ZZT_BBITS>                  constants;

constexpr static uint32_t mask_int_ycoord = constants::mask_int_ycoord;
constexpr static uint32_t width = PARALLELVIEW_ZOOM_WIDTH;

/*****************************************************************************
 * some copy/past of private/local code O:-)
 */
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

static inline uint32_t compute_sec_coord_count_y2(const uint32_t t1,
                                                  const uint32_t t2,
                                                  const uint64_t y_min,
                                                  const uint64_t y_lim,
                                                  const int shift,
                                                  const uint32_t mask,
                                                  const int zoom,
                                                  const uint32_t width,
                                                  const float beta)
{
	bci_code_t bci_min, bci_max;
	uint32_t y2_count;

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * t1,
	                          (uint64_t)BUCKET_ELT_COUNT * t2,
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_min);

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * (t1 + 1),
	                          (uint64_t)BUCKET_ELT_COUNT * (t2 + 1),
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_max);

	if (bci_max.s.type == bci_code_t::UP) {
		// whole top side
		y2_count = PVCore::upper_power_of_2(bci_max.s.r - bci_min.s.r);
	} else if (bci_min.s.type == bci_code_t::DOWN) {
		// whole bottom side
		y2_count = PVCore::upper_power_of_2(bci_min.s.r - bci_max.s.r);
	} else if ((bci_min.s.type == bci_code_t::STRAIGHT)
	           &&
	           (bci_max.s.type == bci_code_t::STRAIGHT)) {
		// opposite side
		y2_count = 1U << PVCore::clamp(zoom, 0, PARALLELVIEW_ZZT_BBITS);
	} else if (bci_min.s.type == bci_code_t::STRAIGHT) {
		// partial bottom side

		// opposite side
		y2_count = constants::image_height - bci_max.s.r;

		// + bottom side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	} else if (bci_max.s.type == bci_code_t::STRAIGHT) {
		// partial top side

		std::cout << "bci: " << std::endl
		          << "  min: " << bci_min.s.type << " " << bci_min.s.l << " " << bci_min.s.r << std::endl
		          << "  max: " << bci_max.s.type << " " << bci_max.s.l << " " << bci_max.s.r << std::endl;
		// opposite side count
		y2_count = bci_min.s.r;

		// + top side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	} else {
		std::cout << "WHOLE" << std::endl;
		// from top side to bottom side

		// opposite side count
		y2_count = constants::image_height;

		// + bottom side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		// + top side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	}

	return PVCore::max(1U, y2_count);
}

/*****************************************************************************
 * input data definition and initialization.
 */
struct data_t
{
	uint32_t *col_a;
	uint32_t *col_b;
	size_t    size;
};

void init_data(data_t &data, size_t num)
{
	data.size = 2048 * num;
	data.col_a = new uint32_t [data.size];
	data.col_b = new uint32_t [data.size];

	size_t idx = 0;
	for(size_t i = 0; i < 2048; ++i) {
		for(size_t j = 0; j < num; ++j) {
			data.col_a[idx] = i;
			data.col_b[idx] = (j / (double)num) * QT_MAX_VALUE;
			++idx;
		}

	}
}

/*****************************************************************************
 * a function to use with valgrind's option '--toggle-collect="
 */
/* to make the next function using C name convention, not C++ one
 */
extern "C"
{
__attribute__((noinline)) void extract(quadtree_t &qt,
                                       uint64_t y1_min, uint64_t y1_max,
                                       uint32_t zoom, uint64_t y2_count,
                                       quadtree_buffer_entry_t *buffer,
                                       const insert_entry_f &insert_f,
                                       tlr_buffer_t &tlr)
{
	qt.get_first_from_y1(y1_min, y1_max, zoom, y2_count,
	                     buffer, insert_f, tlr);

}

}


/*****************************************************************************
 * last one
 */
void usage(const char *program)
{
	std::cerr << "usage: " << basename(program) << " depth num zoom\n" << std::endl;

	std::cerr << "\tdepth: max depth of quadtree" << std::endl;
	std::cerr << "\tnum  : number of events for each primary coordinate event" << std::endl;
	std::cerr << "\tzoom : zoom level in [0,21]" << std::endl;
	std::cerr << std::endl;
	std::cerr << "use option --toggle-collect=extract with valgrind to observ extraction" << std::endl;

}

/*****************************************************************************
 * pouet
 */
int main(int argc, char **argv)
{
	data_t data;

	if (argc != 4) {
		usage(argv[0]);
		exit(1);
	}

	/* data initialization
	 */
	int depth = (size_t) atol(argv[1]);
	size_t num = (size_t) atol(argv[2]);

	BENCH_START(init);
	init_data(data, num);
	BENCH_END(init, "data init", 0, 0, data.size, sizeof(uint32_t) * 2);

	std::cout << "using " << data.size << " events" << std::endl;

	/* quadtree initialization
	 */
	quadtree_t *qt = new quadtree_t(0U, QT_MAX_VALUE, 0U, QT_MAX_VALUE, depth);

	BENCH_START(insert);
	for(size_t i = 0; i < data.size; ++i) {
		qt->insert(quadtree_entry_t(data.col_a[i], data.col_b[i], i));
	}
	BENCH_END(insert, "insertion",
	          data.size, sizeof(uint32_t) * 2,
	          data.size, sizeof(quadtree_entry_t));

	std::cout << "memory before ::compact: " << qt->memory() << std::endl;
	qt->compact();
	std::cout << "memory after ::compact: " << qt->memory() << std::endl;

	quadtree_buffer_entry_t *buffer = new quadtree_buffer_entry_t [QUADTREE_BUFFER_SIZE];
	tlr_buffer_t *tlr = new tlr_buffer_t;

	uint32_t zoom = (uint32_t) atol(argv[3]);

	if (zoom > 21) {
		std::cerr << "zoom too high, using 21" << std::endl;
		zoom = 21;
	}

	/* quadtree extraction's initialization :-P
	 */
	uint64_t y1_min = 0;
	uint64_t y1_max = 1UL << (32 - zoom);
	uint64_t y1_lim = y1_max;
	std::cout << "y1_min  : " << y1_min << std::endl;
	std::cout << "y1_max  : " << y1_max << std::endl;
	std::cout << "y1_lim  : " << y1_lim << std::endl;

	const uint32_t shift = (32 - PARALLELVIEW_ZZT_BBITS) - zoom;
	double beta = 1. / get_scale_factor(zoom);
	std::cout << "shift   : " << shift << std::endl;
	std::cout << "beta    : " << beta << std::endl;

	uint64_t y2_count = compute_sec_coord_count_y2(0, 0,
	                                               y1_min, y1_lim,
	                                               shift, mask_int_ycoord,
	                                               zoom, width, beta);

	std::cout << "zoom    : " << zoom << std::endl;
	std::cout << "y2_count: " << y2_count << std::endl;

	const insert_entry_f insert_f =
		insert_entry_f([&](const quadtree_entry_t &e, tlr_buffer_t &buffer)
		               {
			               bci_code_t bci;
			               compute_bci_projection_y2(e.y1, e.y2,
			                                         y1_min, y1_lim,
			                                         shift, mask_int_ycoord,
			                                         width, beta, bci);
			               tlr_index_t tlr(bci.s.type,
			                               bci.s.l,
			                               bci.s.r);
			               if (e.idx < buffer[tlr.v]) {
				               buffer[tlr.v] = e.idx;
			               }
		               });

	/* quadtree extraction
	 */
	quadtree_t::all_clear();
	quadtree_t::all_count_clear();
	quadtree_t::insert_count_clear();
	BENCH_START(extract);
	extract(*qt,
	        y1_min, y1_max, zoom, y2_count,
	        buffer, insert_f, *tlr);
	BENCH_STOP(extract);

	size_t bci_num = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr)[i] != UINT32_MAX) {
			++bci_num;
		}
	}

	BENCH_SHOW(extract, "extraction",
	           data.size, sizeof(quadtree_entry_t),
	           bci_num, sizeof(uint32_t));

	double all_dt = quadtree_t::all_get();
	std::cout << "QT::visit_y1::all time    : " << all_dt  * 1000. << " ms." << std::endl;
	std::cout << "QT::visit_y1::all count   : " << quadtree_t::all_count_get() << " events." << std::endl;
	std::cout << "QT::visit_y1::test count: " << quadtree_t::test_count_get() << " events." << std::endl;
	std::cout << "QT::visit_y1::insert count: " << quadtree_t::insert_count_get() << " events." << std::endl;

	return 0;
}







