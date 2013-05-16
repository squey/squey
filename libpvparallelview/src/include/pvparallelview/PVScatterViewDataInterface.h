/**
 * \file PVScatterViewDataInterface.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEWDATAINTERFACE_H__
#define __PVSCATTERVIEWDATAINTERFACE_H__


#include <boost/noncopyable.hpp>

#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/PVZoomedZoneTree.h>


namespace PVParallelView {

class PVScatterViewDataInterface : boost::noncopyable
{
public:
	PVScatterViewDataInterface() {};
	virtual ~PVScatterViewDataInterface() {};

public:
	struct ProcessParams
	{
		ProcessParams(
			PVZoomedZoneTree const& zzt,
			const PVCore::PVHSVColor* colors
		) :
			zzt(zzt),
			colors(colors),
			y1_min(0),
			y1_max(0),
			y2_min(0),
			y2_max(0),
			zoom(0),
			alpha(1.0)
		{}

		PVZoomedZoneTree const& zzt;
		const PVCore::PVHSVColor* colors;
		uint64_t y1_min;
		uint64_t y1_max;
		uint64_t y2_min;
		uint64_t y2_max;
		int zoom;
		double alpha;
	};

public:
	virtual void process_bg(ProcessParams const& params) = 0;
	virtual void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) = 0;
	virtual void process_all(ProcessParams const& params, Picviz::PVSelection const& sel)
	{
		process_bg(params);
		process_sel(params, sel);
	}

public:
	//void shift_left(const uint32_t nblocks, const double alpha);
	//void shift_right(const uint32_t nblocks, const double alpha);

public:
	PVScatterViewImage const& image_all() const { return _image_all; }
	PVScatterViewImage const& image_sel() const { return _image_sel; }

	PVScatterViewImage& image_all() { return _image_all; }
	PVScatterViewImage& image_sel() { return _image_sel; }

	void clear()
	{
		image_all().clear();
		image_sel().clear();
	}

private:
	PVScatterViewImage _image_all;
	PVScatterViewImage _image_sel;
};

}


#endif // __PVSCATTERVIEWDATAINTERFACE_H__
