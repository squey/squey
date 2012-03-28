#ifndef __BENCH_H
#define __BENCH_H

#include <tbb/tick_count.h>
#include <iostream>

#define BENCH_START(var)\
	tbb::tick_count __bench_start_##var = tbb::tick_count::now();
#define BENCH_END(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)\
	tbb::tick_count __bench_end_##var = tbb::tick_count::now();\
	{\
		double time = (__bench_end_##var-__bench_start_##var).seconds();\
		double size_in_mb = (double)(nelts_start*size_elt_start)/(1024.0*1024.0);\
		double size_out_mb = (double)(nelts_end*size_elt_end)/(1024.0*1024.0);\
		double bw_in = size_in_mb/time;\
		double bw_out = size_out_mb/time;\
		std::cout << desc << ": in " << time*1000 << " ms. Input (#/size/BW): " << nelts_start << "/" << size_in_mb << " MB/" << bw_in << " MB/s | Output (#/size/BW): " << nelts_end << "/" << size_out_mb << " MB/" << bw_out << " MB/s" << std::endl;\
	}

#define BENCH_END_SAME_TYPE(var, desc, nelts_in, nelts_out, size_elt) BENCH_END(var, desc, nelts_in, size_elt, nelts_out, size_elt)
#define BENCH_END_TRANSFORM(var, desc, nelts, size_elt) BENCH_END_SAME_TYPE(var, desc, nelts, nelts, size_elt)

#define CHECK(v) __CHECK(v, __FILE__, __LINE__)
#define __CHECK(v,F,L)\
	if (!(v)) {\
		std::cerr << F << ":" << L << " :" << #v << " isn't valid." << std::endl;\
	}

#endif
