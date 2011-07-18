//! \file PVColorGradientDualSliderType.h
//! $Id: PVColorGradientDualSliderType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
#define PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H

#include <pvcore/general.h>

namespace PVCore {

/**
 * \class PVColorGradientDualSliderType
 */
class PVColorGradientDualSliderType
{
public:
	PVColorGradientDualSliderType() { _sliders_positions[0] = 0; _sliders_positions[1] = 1; };
	PVColorGradientDualSliderType(const float positions[2]) { set_positions(positions); }

	inline const float* get_positions() const { return _sliders_positions; }
	inline void set_positions(const float pos[2]) { _sliders_positions[0]= pos[0]; _sliders_positions[1] = pos[1]; }

protected:
	float _sliders_positions[2];
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVColorGradientDualSliderType)

#endif // PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
