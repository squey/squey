/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVPredefinedHSVColors.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVHSVColor.h> // for PVHSVColor, HSV_COLOR_BLACK, etc

#include <QSettings>
#include <QString>
#include <QVariant>

constexpr const char* COLORS_SETTINGS_ARRAY = "predefined_colors";

std::vector<PVCore::PVHSVColor> PVCore::PVPredefinedHSVColors::get_predefined_colors()
{
	std::vector<PVCore::PVHSVColor> ret;
	ret.resize(get_predefined_colors_count(), HSV_COLOR_WHITE);

	QSettings& pvconfig = PVCore::PVConfig::get().config();

	pvconfig.beginGroup(COLORS_SETTINGS_ARRAY);
	for (size_t i = 0; i < get_predefined_colors_count(); i++) {
		QString istr = QString::number(i);
		if (pvconfig.contains(istr)) {
			PVCore::PVHSVColor c(pvconfig.value(istr).toInt());
			if (!c.is_valid() || c == HSV_COLOR_BLACK) {
				c = HSV_COLOR_WHITE;
			}
			ret[i] = c;
		}
	}
	pvconfig.endGroup();

	return ret;
}

bool PVCore::PVPredefinedHSVColors::set_predefined_color(size_t i, PVCore::PVHSVColor c)
{
	if (!c.is_valid() || c == HSV_COLOR_BLACK) {
		return false;
	}

	QSettings& pvconfig = PVCore::PVConfig::get().config();

	pvconfig.beginGroup(COLORS_SETTINGS_ARRAY);
	pvconfig.setValue(QString::number(i), QVariant(c.h()));
	pvconfig.endGroup();

	return true;
}
