
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPredefinedHSVColors.h>
#include <pvkernel/core/PVConfig.h>

#define COLORS_SETTINGS_ARRAY "predefined_colors"

std::vector<PVCore::PVHSVColor> PVCore::PVPredefinedHSVColors::get_predefined_colors()
{
	std::vector<PVCore::PVHSVColor> ret;
	ret.resize(get_predefined_colors_count(), PVCore::PVHSVColor(HSV_COLOR_WHITE));

	QSettings &pvconfig = PVCore::PVConfig::get().config();

	pvconfig.beginGroup(COLORS_SETTINGS_ARRAY);
	for (size_t i = 0; i < get_predefined_colors_count(); i++) {
		QString istr = QString::number(i);
		if (pvconfig.contains(istr)) {
			PVCore::PVHSVColor c(pvconfig.value(istr).toInt());
			if (!c.is_valid() || c.h() == HSV_COLOR_BLACK) {
				c.h() = HSV_COLOR_WHITE;
			}
			ret[i] = c;
		}
	}
	pvconfig.endGroup();

	return std::move(ret);
}

bool PVCore::PVPredefinedHSVColors::set_predefined_color(size_t i, PVCore::PVHSVColor c)
{
	if (!c.is_valid() || c.h() == HSV_COLOR_BLACK) {
		return false;
	}

	QSettings &pvconfig = PVCore::PVConfig::get().config();

	pvconfig.beginGroup(COLORS_SETTINGS_ARRAY);
	pvconfig.setValue(QString::number(i), QVariant(c.h()));
	pvconfig.endGroup();

	return true;
}
