/**
 * \file PVHardwareConcurrency.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVHARDWARECONCURRENCY_H_
#define PVHARDWARECONCURRENCY_H_

#include <hwloc.h>

namespace PVCore
{
class PVHardwareConcurrency
{
public:
	static inline int get_physical_core_number()
	{
		return get()._nb_cores;
	}

	static inline int get_logical_core_number()
	{
		return get()._nb_threads;
	}

	static inline bool is_hyperthreading_enabled()
	{
		return get_logical_core_number() >= 2*get_physical_core_number();
	}

private:
	static PVHardwareConcurrency& get()
	{
		if (_hardware_concurrency == nullptr) {
			_hardware_concurrency = new PVHardwareConcurrency;
		}
		return *_hardware_concurrency;
	}

	PVHardwareConcurrency()
	{
	    int depth;
	    hwloc_topology_t topology;

	    // init hwloc topology
	    hwloc_topology_init(&topology);
	    hwloc_topology_load(topology);

	    _nb_cores = 0;
	    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
	    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
	    	_nb_cores = hwloc_get_nbobjs_by_depth(topology, depth);
	    }

	    _nb_threads = 0;
	    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
	    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
	    	_nb_threads = hwloc_get_nbobjs_by_depth(topology, depth);
	    }

	    // destroy hwloc topology
	    hwloc_topology_destroy(topology);
	}

private:
	static PVHardwareConcurrency* _hardware_concurrency;

	uint32_t _nb_cores;
	uint32_t _nb_threads;
};
}

#endif /* PVHARDWARECONCURRENCY_H_ */
