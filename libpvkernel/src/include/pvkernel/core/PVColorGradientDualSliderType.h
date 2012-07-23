/**
 * \file PVColorGradientDualSliderType.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
#define PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

namespace PVCore {

/**
 * \class PVColorGradientDualSliderType
 */
class PVColorGradientDualSliderType: public PVArgumentType<PVColorGradientDualSliderType>
{
public:
	PVColorGradientDualSliderType() { _sliders_positions[0] = 0; _sliders_positions[1] = 1; };
	PVColorGradientDualSliderType(const float positions[2]) { set_positions(positions); }

	inline const float* get_positions() const { return _sliders_positions; }
	inline void set_positions(const float pos[2]) { _sliders_positions[0]= pos[0]; _sliders_positions[1] = pos[1]; }

	QString to_string() const
	{
		return QString::number(_sliders_positions[0]) + "," + QString::number(_sliders_positions[1]);
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		PVArgument arg;
		bool ok1 = false;
		bool ok2 = false;

		QStringList strList = str.split(",");
		if (strList.count() == 2) {
			float pos[2] = {strList[0].toFloat(&ok1), strList[1].toFloat(&ok2)};
			arg.setValue(PVColorGradientDualSliderType(pos));
		}

		if (ok) {
			*ok = ok1 && ok2;
		}

		return arg;

	}
	bool operator==(const PVColorGradientDualSliderType &other) const
	{
		return _sliders_positions[0] == other._sliders_positions[0] &&
			   _sliders_positions[1] == other._sliders_positions[1] ;
	}

protected:
	float _sliders_positions[2];
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVColorGradientDualSliderType)

#endif // PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
