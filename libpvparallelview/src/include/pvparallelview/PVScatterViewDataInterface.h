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
#include <pvparallelview/PVScatterViewImage.h>


namespace PVParallelView {

class PVScatterViewDataInterface : boost::noncopyable
{
public:
	PVScatterViewDataInterface() {};
	virtual ~PVScatterViewDataInterface() {};

public:
	struct ProcessParams
	{
		struct dirty_rect
		{
			dirty_rect() : y1_min(0), y1_max(0), y2_min(0), y2_max(0) {}
			uint64_t y1_min;
			uint64_t y1_max;
			uint64_t y2_min;
			uint64_t y2_max;
		};

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
			alpha(1.0),
			y1_offset(0),
			y2_offset(0)
		{}

		dirty_rect rect_1() const;
		dirty_rect rect_2() const;
		int32_t map_to_view(int64_t scene_value) const;
		QRect map_to_view(const dirty_rect& rect) const;

		PVZoomedZoneTree const& zzt;
		const PVCore::PVHSVColor* colors;
		uint64_t y1_min;
		uint64_t y1_max;
		uint64_t y2_min;
		uint64_t y2_max;
		int zoom;
		double alpha;
		int64_t y1_offset;
		int64_t y2_offset;
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
	PVScatterViewImage const& image_all() const { return _image_all; }
	PVScatterViewImage const& image_sel() const { return _image_sel; }

	PVScatterViewImage& image_all() { return _image_all; }
	PVScatterViewImage& image_sel() { return _image_sel; }

private:
	PVScatterViewImage _image_all;
	PVScatterViewImage _image_sel;
};

}


#endif // __PVSCATTERVIEWDATAINTERFACE_H__
