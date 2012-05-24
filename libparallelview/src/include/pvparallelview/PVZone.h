#ifndef PVPARALLELVIEW_PVZONE_H
#define PVPARALLELVIEW_PVZONE_H

#include <pvparallelview/PVZoneTree.h>

namespace PVParallelView {

class PVZone
{
public:
	PVZone():
		_ztree(new PVZoneTree()),
		_width(PVParallelView::ZoneDefaultWidth)
	{ }

public:
	inline void set_width(uint32_t width) const { _width = width; }
	uint32_t width() const { return _width; }

	PVZoneTree& ztree() { return *_ztree; }
	PVZoneTree const& ztree() const { return *_ztree; }

	template <class Tree>
	Tree const& get_tree() const
	{
		assert(false);
		return Tree();
	}

private:
	PVZoneTree_p _ztree;
	mutable uint32_t _width; // TODO: Fix that
};

template <>
inline PVZoneTree const& PVZone::get_tree<PVZoneTree>() const
{
	return *_ztree;
}

}

#endif
