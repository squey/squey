/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVCOLOR_H
#define PVCORE_PVCOLOR_H

#include <cstdint>

#include <QColor>

namespace PVCore
{

/**
 * \struct PVColor
 */
struct PVColor {

	QColor toQColor() const;
	void fromQColor(QColor const& color);

	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	uint8_t a = 255;
};
}

#endif /* INENDI_PVCOLOR_H */
