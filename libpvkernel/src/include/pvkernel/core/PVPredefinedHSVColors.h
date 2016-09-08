/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVPREDEFINED_HSV_COLORS_H
#define PVCORE_PVPREDEFINED_HSV_COLORS_H

#include <pvkernel/core/PVHSVColor.h>

#include <vector>
#include <cstddef>

namespace PVCore
{

class PVPredefinedHSVColors
{
  public:
	static inline constexpr size_t get_predefined_colors_count() { return 22; }

	static std::vector<PVCore::PVHSVColor> get_predefined_colors();
	static bool set_predefined_color(size_t i, PVCore::PVHSVColor c);
};
}

#endif
