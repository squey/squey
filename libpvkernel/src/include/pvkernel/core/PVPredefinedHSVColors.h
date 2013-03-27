#ifndef PVCORE_PVPREDEFINED_HSV_COLORS_H
#define PVCORE_PVPREDEFINED_HSV_COLORS_H

#include <pvkernel/core/PVHSVColor.h>

#include <vector>

namespace PVCore {

class PVPredefinedHSVColors
{
public:
	static inline constexpr size_t get_predefined_colors_count() { return 16; }

	static std::vector<PVCore::PVHSVColor> get_predefined_colors();
	static bool set_predefined_color(size_t i, PVCore::PVHSVColor c);
};

}

#endif
