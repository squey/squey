/**
 * \file PVHardwareConcurrency.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVHARDWARECONCURRENCY_H_
#define PVHARDWARECONCURRENCY_H_

#include <vector>

#include <hwloc.h>

namespace PVCore
{

/**
 * \class PVHardwareConcurrency
 *
 * \note This class gives hardware information using hwloc library.
 *
 */
class PVHardwareConcurrency
{
public:
	/*! \brief Returns the number of physical cores.
	 *
	 *  \return Physical core number.
	 */
	static inline uint32_t get_physical_core_number()
	{
		return get()._nb_cores;
	}

	/*! \brief Returns the number of logical cores.
	 *
	 *  \note a logical core -or thread- usually shares all ressources but registers.
	 *
	 *  \return Physical logical number.
	 */
	static inline uint32_t get_logical_core_number()
	{
		return get()._nb_threads;
	}

	/*! \brief Check if hyperthreading is enabled.
	 *
	 *  \return True if hyperthreading is enabled, false otherwise.
	 */
	static inline bool is_hyperthreading_enabled()
	{
		return get_logical_core_number() >= 2*get_physical_core_number();
	}

	/*! \brief Returns the number of cache levels.
	 *
	 *  \return The number of cache levels.
	 */
	static inline size_t get_cache_levels()
	{
		return get()._cache_level_sizes.size();
	}

	/*! \brief Returns the size of a cache located at a given level.
	 *
	 *  \param[in] level_n The cache level. First cache is located at index 0.
	 *
	 *  \return The size of the cache if it exists, 0 otherwise.
	 */
	static inline size_t get_level_n_cache_size(uint32_t level_n)
	{
		if (level_n < get()._cache_level_sizes.size()) {
			return get()._cache_level_sizes[level_n];
		}

		return 0;
	}

private:
	static PVHardwareConcurrency& get()
	{
		if (_hardware_concurrency == nullptr) {
			_hardware_concurrency = new PVHardwareConcurrency();
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

	    // physical cores
	    _nb_cores = 0;
	    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
	    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
	    	_nb_cores = hwloc_get_nbobjs_by_depth(topology, depth);
	    }

	    // logical cores
	    _nb_threads = 0;
	    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
	    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
	    	_nb_threads = hwloc_get_nbobjs_by_depth(topology, depth);
	    }

	    // caches sizes
	    hwloc_obj_t obj;
	    for (obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0); obj; obj = obj->parent) {
			if (obj->type == HWLOC_OBJ_CACHE) {
				_cache_level_sizes.push_back(obj->attr->cache.size);
			}
	    }

	    // destroy hwloc topology
	    hwloc_topology_destroy(topology);
	}

private:
	static PVHardwareConcurrency* _hardware_concurrency;

	uint32_t _nb_cores;
	uint32_t _nb_threads;
	std::vector<size_t> _cache_level_sizes;
};
}

#endif /* PVHARDWARECONCURRENCY_H_ */
