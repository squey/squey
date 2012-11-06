/**
 * \file PVZone.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONE_H
#define PVPARALLELVIEW_PVZONE_H

#include <tbb/atomic.h>

#include <pvkernel/core/PVTypeTraits.h>

#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/common.h>

namespace PVParallelView {

class PVZone
{
public:
	PVZone():
		_ztree(new PVZoneTree()),
		_zoomed_ztree(new PVZoomedZoneTree(_ztree->get_sel_elts()))
	{ }

public:
	PVZoneTree& ztree() { return *_ztree; }
	PVZoneTree const& ztree() const { return *_ztree; }

	PVZoomedZoneTree& zoomed_ztree() { return *_zoomed_ztree; }
	PVZoomedZoneTree const& zoomed_ztree() const { return *_zoomed_ztree; }

	inline void filter_by_sel(const Picviz::PVSelection& sel, const PVRow nrows)
	{
		_ztree->filter_by_sel(sel, nrows);
	}

	inline void filter_by_sel_background(const Picviz::PVSelection& sel, const PVRow nrows)
	{
		_ztree->filter_by_sel_background(sel, nrows);
	}

	template <class Tree>
	Tree const& get_tree() const
	{
		assert(false);
		return *(new Tree());
	}

	template <class Tree>
	Tree& get_tree()
	{
		assert(false);
		return *(new Tree());
	}

	void reset()
	{
		_ztree.reset(new PVZoneTree());
		_zoomed_ztree.reset(new PVZoomedZoneTree(_ztree->get_sel_elts()));
	}

private:
	PVZoneTree_p _ztree;
	PVZoomedZoneTree_p _zoomed_ztree;

};

template <>
inline PVZoneTree const& PVZone::get_tree<PVZoneTree>() const
{
	return *_ztree;
}

template <>
inline PVZoneTree& PVZone::get_tree<PVZoneTree>()
{
	return *_ztree;
}

template <>
inline PVZoomedZoneTree const& PVZone::get_tree<PVZoomedZoneTree>() const
{
	return *_zoomed_ztree;
}

template <>
inline PVZoomedZoneTree& PVZone::get_tree<PVZoomedZoneTree>()
{
	return *_zoomed_ztree;
}

}

#endif
