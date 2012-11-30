/**
 * \file picviz_bench.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __BENCH_H
#define __BENCH_H

#ifdef PICVIZ_DEVELOPER_MODE

#include <tbb/tick_count.h>
#include <iostream>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/picviz_stat.h>

#define BENCH_START(var)\
	tbb::tick_count __bench_start_##var = tbb::tick_count::now();
#define BENCH_STOP(var)\
	tbb::tick_count __bench_end_##var = tbb::tick_count::now();
#define BENCH_END_NODISP(var)\
	BENCH_STOP(var);
#define BENCH_SHOW(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)\
	{\
		double time = (__bench_end_##var-__bench_start_##var).seconds();\
		double size_in_mb = (double)(nelts_start*size_elt_start)/(1024.0*1024.0);\
		double size_out_mb = (double)(nelts_end*size_elt_end)/(1024.0*1024.0);\
		double bw_in = size_in_mb/time;\
		double bw_out = size_out_mb/time;\
		std::cout << desc << ": in " << time*1000 << " ms. Input (#/size/BW): " << nelts_start << "/" << size_in_mb << " MB/" << bw_in << " MB/s | Output (#/size/BW): " << nelts_end << "/" << size_out_mb << " MB/" << bw_out << " MB/s" << std::endl;\
	}

#define BENCH_END(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)\
	BENCH_STOP(var);\
	BENCH_SHOW(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end);

#define BENCH_STAT_TIME(var) \
	{\
		double time = (__bench_end_##var-__bench_start_##var).seconds();\
		PV_STAT_TIME_MSEC(#var "_time", time*1000); \
	}

#define BENCH_STAT_IN_BW(var, nelts_in, size_elt_in) \
	{\
		double time = (__bench_end_##var-__bench_start_##var).seconds();\
		double size_in_mb = (double)(nelts_in*size_elt_in)/(1024.0*1024.0);\
		double bw_in = size_in_mb/time;\
		PV_STAT_MEM_BW(#var "_in_bw", bw_in); \
	}

#define BENCH_STAT_OUT_BW(var, nelts_out, size_elt_out) \
	{\
		double time = (__bench_end_##var-__bench_start_##var).seconds();\
		double size_out_mb = (double)(nelts_out*size_elt_out)/(1024.0*1024.0);\
		double bw_out = size_out_mb/time;\
		PV_STAT_MEM_BW(#var "_out_bw", bw_out); \
	}

#define BENCH_STAT(var, nelts_in, size_elt_in, nelts_out, size_elt_out)  \
	BENCH_STAT_TIME(var); \
	BENCH_STAT_IN_BW(var, nelts_in, size_elt_in); \
	BENCH_STAT_OUT_BW(var, nelts_out, size_elt_out)

#define BENCH_END_TIME(var)\
		((__bench_end_##var-__bench_start_##var).seconds())

#define BENCH_END_SAME_TYPE(var, desc, nelts_in, nelts_out, size_elt) BENCH_END(var, desc, nelts_in, size_elt, nelts_out, size_elt)
#define BENCH_END_TRANSFORM(var, desc, nelts, size_elt) BENCH_END_SAME_TYPE(var, desc, nelts, nelts, size_elt)

// Memory benchmarking
#define MEM_START(var)\
	double __mem_bench_##var_vm, __mem_bench_##var_rss;\
	PVCore::PVMemory::get_memory_usage(__mem_bench_##var_vm, __mem_bench_##var_rss);
#define MEM_END(var, desc)\
	double __mem_bench_##var_vm_end, __mem_bench_##var_rss_end;\
	PVCore::PVMemory::get_memory_usage(__mem_bench_##var_vm_end, __mem_bench_##var_rss_end);\
	double diff_vm_mb = (__mem_bench_##var_vm_end-__mem_bench_##var_vm)/(1024.0);\
	double diff_rss_mb = (__mem_bench_##var_rss_end-__mem_bench_##var_rss)/(1024.0);\
	std::cout << desc << ": memory footprint is: VM=" << diff_vm_mb << " MB / RES=" << diff_rss_mb << " MB" << std::endl;

#else

#define BENCH_START(var)
#define BENCH_STOP(var)
#define BENCH_END_NODISP(var)
#define BENCH_SHOW(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)
#define BENCH_END(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)
#define BENCH_STAT_TIME(var)
#define BENCH_STAT_IN_BW(var, nelts_in, size_elt_start, in)
#define BENCH_STAT_OUT_BW(var, nelts_out, size_elt_out)
#define BENCH_STAT(var, nelts_in, size_elt_in, nelts_out, size_elt_out)
#define BENCH_END_TIME(var) (0)
#define BENCH_END_SAME_TYPE(var, desc, nelts_in, nelts_out, size_elt) BENCH_END(var, desc, nelts_in, size_elt, nelts_out, size_elt)
#define BENCH_END_TRANSFORM(var, desc, nelts, size_elt) BENCH_END_SAME_TYPE(var, desc, nelts, nelts, size_elt)
#define MEM_START(var)
#define MEM_END(var, desc)

#endif

#define CHECK(v) __CHECK(v, __FILE__, __LINE__)
#define __CHECK(v,F,L)\
	if (!(v)) {\
		std::cerr << F << ":" << L << " :" << #v << " isn't valid." << std::endl;\
	}

#endif
