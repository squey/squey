/**
 * \file PVScatterViewImagesManager.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEWIMAGESMANAGER_H__
#define __PVSCATTERVIEWIMAGESMANAGER_H__

#include <boost/utility.hpp>

#include <pvparallelview/common.h>

#include <pvparallelview/PVScatterViewImage.h>
#include <pvparallelview/PVScatterViewData.h>

#include <pvparallelview/PVZoneRenderingScatter_types.h>

namespace PVParallelView
{

class PVZonesProcessor;

class PVScatterViewImagesManager: boost::noncopyable
{
protected:
	typedef PVScatterViewData::ProcessParams DataProcessParams;

public:
	PVScatterViewImagesManager(
		PVZoneID const zid,
		PVZonesProcessor& zp_bg,
		PVZonesProcessor& zp_sel,
		PVZoomedZoneTree const& zzt,
		const PVCore::PVHSVColor* colors,
		Picviz::PVSelection const& sel
	);

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

public:
	inline uint64_t last_y1_min() const { return _data_params.y1_min; }
	inline uint64_t last_y1_max() const { return _data_params.y1_max; }
	inline uint64_t last_y2_min() const { return _data_params.y2_min; }
	inline uint64_t last_y2_max() const { return _data_params.y2_max; }
	inline int last_zoom() const { return _data_params.zoom; }
	inline double last_alpha() const { return _data_params.alpha; }
	inline bool params_changed(
		const uint64_t y1_min,
		const uint64_t y1_max,
		const uint64_t y2_min,
		const uint64_t y2_max,
		const int zoom,
		const double alpha) const
	{
		return !(y1_min == last_y1_min() &&
				 y1_max == last_y1_max() &&
				 y2_min == last_y2_min() &&
				 y2_max == last_y2_max() &&
				 zoom == last_zoom() &&
				 alpha == last_alpha());
	}

	inline void set_img_update_receiver(QObject* obj) { _img_update_receiver = obj; }

protected:
	inline void set_params(
		const uint64_t y1_min,
		const uint64_t y1_max,
		const uint64_t y2_min,
		const uint64_t y2_max,
		const int zoom,
		const double alpha
	)
	{
		_data_params.y1_min = y1_min;
		_data_params.y1_max = y1_max;
		_data_params.y2_min = y2_min;
		_data_params.y2_max = y2_max;
		_data_params.zoom = zoom;
		_data_params.alpha = alpha;
	}

	inline bool full_view() const { return (_data_params.zoom == 0) && (_data_params.alpha == 1.0); }

	void connect_zr(PVZoneRenderingScatter& zr, const char* slot);

protected:
	PVZoneID _zid;

	PVScatterViewData _data_z0; // Data for initial zoom (with 10-bit precision)
	PVScatterViewData _data;

	Picviz::PVSelection const& _sel;

	DataProcessParams _data_params;

	PVZoneRenderingScatter_p _zr_bg;
	PVZoneRenderingScatter_p _zr_sel;

	PVZonesProcessor& _zp_bg;
	PVZonesProcessor& _zp_sel;

	PVScatterViewImage _last_image_all;
	PVScatterViewImage _last_image_bg;

	QObject* _img_update_receiver;
};

}

#endif // __PVSCATTERVIEWIMAGESMANAGER_H__
