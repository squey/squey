//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
