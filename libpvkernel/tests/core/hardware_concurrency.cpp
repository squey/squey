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
	PVLOG_INFO("Number of physical core(s): %d\n", nb_cores);
	PVLOG_INFO("Number of logical core(s): %d\n", nb_threads);
	PVLOG_INFO("Is HyperThreading enabled: %d\n", hyperthreading);
}
