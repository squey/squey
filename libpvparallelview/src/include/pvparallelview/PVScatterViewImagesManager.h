/**
 * \file PVScatterViewImagesManager.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEWIMAGESMANAGER_H__
#define __PVSCATTERVIEWIMAGESMANAGER_H__

#include <boost/utility.hpp>

#include <pvparallelview/PVScatterViewImage.h>
#include <pvparallelview/PVScatterViewData.h>

namespace PVParallelView
{

class PVScatterViewImagesManager : boost::noncopyable
{
protected:
	typedef PVScatterViewData::ProcessParams DataProcessParams;

public:
	PVScatterViewImagesManager(
		PVZoomedZoneTree const& zzt,
		const PVCore::PVHSVColor* colors,
		Picviz::PVSelection const& sel
	) : _sel(sel), _data_params(zzt, colors) {};

public:
	bool change_and_process_view(
		const uint64_t y1_min,
		const uint64_t y1_max,
		const uint64_t y2_min,
		const uint64_t y2_max,
		const int zoom,
		const double alpha
	);

public:
	void process_bg();
	void process_sel();
	void process_all();

public:
	const QImage& get_image_sel();
	const QImage& get_image_all();

protected:
	inline bool full_view() const { return (_data_params.zoom == 0) && (_data_params.alpha == 1.0); }

protected:
	PVScatterViewData _data_z0; // Data for initial zoom (with 10-bit precision)
	PVScatterViewData _data;

	Picviz::PVSelection const& _sel;

	DataProcessParams _data_params;
};

}

#endif // __PVSCATTERVIEWIMAGESMANAGER_H__
