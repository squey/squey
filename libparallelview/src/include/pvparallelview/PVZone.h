#ifndef PVPARALLELVIEW_PVZONE_H
#define PVPARALLELVIEW_PVZONE_H

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
		_zoomed_ztree(new PVZoomedZoneTree()),
		_width(PVParallelView::ZoneDefaultWidth)
	{ }

public:
	inline void set_width(uint32_t width) { assert(width <= PVParallelView::ZoneMaxWidth); _width = width; }
	uint32_t width() const { return _width; }

	PVZoneTree& ztree() { return *_ztree; }
	PVZoneTree const& ztree() const { return *_ztree; }

	PVZoomedZoneTree& zoomed_ztree() { return *_zoomed_ztree; }
	PVZoomedZoneTree const& zoomed_ztree() const { return *_zoomed_ztree; }

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

private:
	PVZoneTree_p _ztree;
	PVZoomedZoneTree_p _zoomed_ztree;
	uint32_t _width;
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
