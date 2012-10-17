/**
 * \file PVMultiGrid.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVMULTIGRID_H
#define PARALLELVIEW_PVMULTIGRID_H

#include <pvbase/types.h>

#include <pvkernel/core/PVVector.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVBCICode.h>

namespace PVParallelView {

#define PVMULTIGRID_MAX_NODE_ELEMENT_COUNT 100000

#pragma pack(push)
#pragma pack(4)

struct PVMultiGridEntry {
	uint32_t y1;
	uint32_t y2;
	PVRow    idx;
};
#pragma pack(pop)

typedef PVCore::PVVector<PVMultiGridEntry> pvmultigrid_entries_t;

template <int ORDER>
class PVMultiGrid
{
public:
	PVMultiGrid(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		uint32_t y1_step = (y1_max_value + 1 - y1_min_value) >> ORDER;
		uint32_t y2_step = (y2_max_value + 1 - y2_min_value) >> ORDER;

		init(y1_min_value, y1_step, y2_min_value, y2_step, max_level);
	}

	~PVMultiGrid()
	{
		if (_nodes == 0) {
			_datas.clear();
		} else {
			delete [] _nodes;
		}
	}

	int max_depth()
	{
		if (_nodes != 0) {
			int depth = -1;
			int d;
			for (int i = 0; i < (1 << ORDER) * (1 << ORDER); ++i) {
				d = _nodes[i].max_depth();
				if (d > depth) {
					depth = d;
				}
			}
			return depth + 1;
		} else {
			return 1;
		}
	}

	inline size_t memory() const
	{
		size_t mem = sizeof (PVMultiGrid) - sizeof(pvmultigrid_entries_t) + _datas.memory();
		if (_nodes != 0) {
			for(int i = 0; i < (1 << ORDER) * (1 << ORDER); ++i) {
				mem += _nodes[i].memory();
			}
		}
		return mem;
	}

	void insert(const PVMultiGridEntry &e) {
		// searching for the right child
		PVMultiGrid *mg = this;
		while (mg->_nodes != 0) {
			int idx = mg->compute_index(e);
			mg = &mg->_nodes[idx];
		}

		// insertion
		mg->_datas.push_back(e);

		// does the current node must be splitted?
		if ((mg->_datas.size() >= PVMULTIGRID_MAX_NODE_ELEMENT_COUNT) && mg->_max_level) {
			mg->create_next_level();
		}
	}

private:
	// CTOR to use with call to init()
	PVMultiGrid()
	{
	}

	void init(uint32_t y1_min_value, uint32_t y1_step_value, uint32_t y2_min_value, uint32_t y2_step_value, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y2_min_value = y2_min_value;
		_y1_step = y1_step_value;
		_y2_step = y2_step_value;
		_max_level = max_level;
		// + 1 to avoid reallocating before a split occurs
		_datas.reserve(PVMULTIGRID_MAX_NODE_ELEMENT_COUNT + 1);
		_nodes = 0;

	}

	inline int compute_index(const PVMultiGridEntry &e) const
	{
		int y1 = (e.y1 - _y1_min_value) / _y1_step;
		int y2 = (e.y2 - _y2_min_value) / _y2_step;
		return (y2 << ORDER) + y1;
	}

	void create_next_level()
	{
		uint32_t y1_min, y2_min;
		uint32_t y1_step = _y1_step >> ORDER;
		uint32_t y2_step = _y2_step >> ORDER;

		_nodes = new PVMultiGrid [(1 << ORDER) * (1 << ORDER)];

		y2_min = _y2_min_value;
		for (int y2 = 0; y2 < (1 << ORDER); ++y2) {
			y1_min = _y1_min_value;
			for (int y1 = 0; y1 < (1 << ORDER); ++y1) {
				_nodes[(y2 << ORDER) + y1].init(y1_min, y1_step,
				                                y2_min, y2_step,
				                                _max_level - 1);
				y1_min += _y1_step;
			}
			y2_min += _y2_step;
		}

		for (unsigned i = 0; i < _datas.size(); ++i) {
			PVMultiGridEntry &e = _datas.at(i);
			_nodes[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

private:
	pvmultigrid_entries_t  _datas;
	PVMultiGrid           *_nodes;

	uint32_t               _y1_min_value;
	uint32_t               _y2_min_value;
	uint32_t               _y1_step;
	uint32_t               _y2_step;
	uint32_t               _max_level;

};

}

#endif // PARALLELVIEW_PVMULTIGRID_H




