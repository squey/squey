#include <picviz/PVTFViewRowFilterFastIPv4.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/picviz_bench.h>

#include <pvkernel/core/network.h>
#include <tbb/parallel_sort.h>

typedef struct {
	uint32_t ip;
	PVRow row;
} IPStruct;

bool IPStruct_lesser(const IPStruct &a, const IPStruct &b)
{
	return a.ip < b.ip;
}

Picviz::PVSelection Picviz::PVTFViewRowFilterFastIPv4::operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const
{
	/* RH: This version is really fast (full selection in 890 ms for 2M lines)
	 * but two big limitations:
	 * - uses mapping space values, not nraw's ones
	 * - no PVSelRowFilteringFunction
	 * So:
	 * 1/ can it be modified to work with nraw values?
	 * 2/ can the merge use N entries instead of just 2?
	 */
	PVSelection sel_ret;
	PVRow nlines_src = view_src.get_row_count();
	std::vector<IPStruct> ips_src;
	std::vector<IPStruct> ips_dst;
	IPStruct p;
	PVRow count_src = 0;
	PVRow count_dst = view_dst.get_row_count();
	PVRow index_src, index_dst;

	BENCH_START(whole);
	BENCH_START(preprocess);
	BENCH_START(preprocess_src);

	BENCH_START(convert_src);
	// extraction of src selected rows
	ips_src.reserve(view_src.get_row_count());
	for(PVRow r = 0; r < nlines_src; ++r) {
		if (!sel_org.get_line(r))
			continue;

		PVCore::Network::ipv4_aton(view_src.get_data_raw(r, 1), p.ip);
		p.row = r;
		ips_src.push_back(p);
	}
	count_src = ips_src.size();
        BENCH_END(convert_src, "convert_src", 1, 1, sizeof(uint32_t), ips_src.size());

	BENCH_START(sort_src);
	// TODO: try sorting with GPU
	tbb::parallel_sort(ips_src.begin(), ips_src.end(), IPStruct_lesser);
        BENCH_END(sort_src, "sort_src", 1, 1, 1, 1);

        BENCH_END(preprocess_src, "preprocess_src", 1, 1, 1, 1);

	BENCH_START(preprocess_dst);

	BENCH_START(convert_dst);
	// extraction of dst selected rows
	ips_dst.resize(count_dst);
#pragma omp parallel for
	for(PVRow r = 0; r < count_dst; ++r) {
		PVCore::Network::ipv4_aton(view_dst.get_data_raw(r, 1), ips_dst[r].ip);
		ips_dst[r].row = r;
	}
	BENCH_END(convert_dst, "convert_dst", 1, 1, sizeof(uint32_t), ips_dst.size());

	BENCH_START(sort_dst);
        tbb::parallel_sort(ips_dst.begin(), ips_dst.end(), IPStruct_lesser);
        BENCH_END(sort_dst, "sort_dst", 1, 1, 1, 1);

        BENCH_END(preprocess_dst, "preprocess_dst", 1, 1, 1, 1);
        BENCH_END(preprocess, "preprocess", 1, 1, 1, 1);

	BENCH_START(merge);

	// time to compute the resulting selection
	index_src = index_dst = 0;
	sel_ret.select_none();

	while((index_src < count_src) && (index_dst < count_dst)) {
		if(ips_src[index_src].ip < ips_dst[index_dst].ip) {
			++index_src;
		} else if(ips_src[index_src].ip > ips_dst[index_dst].ip) {
			++index_dst;
		} else {
			uint32_t &value = ips_src[index_src].ip;
			while ((value == ips_dst[index_dst].ip) && (index_dst < count_dst)) {
				sel_ret.set_line(ips_dst[index_dst].row, true);
				++index_dst;
			}
			while ((value == ips_src[index_src].ip) && (index_src < count_src)) {
				++index_src;
			}
		}
	}

        BENCH_END(merge, "merge", 1, 1, 1, 1);
        BENCH_END(whole, "whole", 1, 1, 1, 1);

	return sel_ret;
}
