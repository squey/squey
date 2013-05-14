/**
 * \file hardware_concurrency.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/general.h>

int main()
{
	uint32_t nb_cores = PVCore::PVHardwareConcurrency::get_physical_core_number();
	uint32_t nb_threads = PVCore::PVHardwareConcurrency::get_logical_core_number();
	bool hyperthreading = PVCore::PVHardwareConcurrency::is_hyperthreading_enabled();
	int32_t nb_cache_levels = PVCore::PVHardwareConcurrency::get_cache_levels();
	PVLOG_INFO("Number of physical core(s): %d\n", nb_cores);
	PVLOG_INFO("Number of logical core(s): %d\n", nb_threads);
	PVLOG_INFO("Is HyperThreading enabled: %d\n", hyperthreading);
	PVLOG_INFO("Number of cache levels: %d\n", nb_cache_levels);
	for (int n = 0; n < nb_cache_levels; n++) {
		PVLOG_INFO("    Level %d cache size : %d\n", n+1, PVCore::PVHardwareConcurrency::get_level_n_cache_size(n));
	}
	PVLOG_INFO("    Level %d cache size: %d\n", -1, PVCore::PVHardwareConcurrency::get_level_n_cache_size(-1));
	PVLOG_INFO("    Level %d cache size : %d\n", nb_cache_levels+1, PVCore::PVHardwareConcurrency::get_level_n_cache_size(nb_cache_levels+1));
}
