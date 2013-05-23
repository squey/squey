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

class PVZonesManager;
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
		PVZonesManager const& zm,
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
	void set_zone(PVZoneID const zid);
	void cancel_all_and_wait();

public:
	const QImage& get_image_sel() const;
	const QImage& get_image_all() const;

	PVZoneID get_zone_index() const { return _zid; }
	PVZonesManager const& get_zones_manager() const { return _zm; }

public:
	/*inline uint64_t last_y1_min() const { return _data_params.y1_min; }
	inline uint64_t last_y1_max() const { return _data_params.y1_max; }
	inline uint64_t last_y2_min() const { return _data_params.y2_min; }
	inline uint64_t last_y2_max() const { return _data_params.y2_max; }
	inline int last_zoom() const { return _data_params.zoom; }
	inline double last_alpha() const { return _data_params.alpha; }*/

	inline void set_img_update_receiver(QObject* obj) { _img_update_receiver = obj; }

private:
	void connect_zr(PVZoneRenderingScatter& zr, const char* slot);

	static void clear_dirty_rects(DataProcessParams const& params, PVScatterViewImage& image);

	void process_bg(DataProcessParams const& params);
	void process_sel(DataProcessParams const& params);

protected:
	PVZoneID _zid;
	PVZonesManager const& _zm;

	PVScatterViewData _data;

	Picviz::PVSelection const& _sel;
	PVCore::PVHSVColor const* _colors;

	PVZoneRenderingScatter_p _zr_bg;
	PVZoneRenderingScatter_p _zr_sel;

	PVZonesProcessor& _zp_bg;
	PVZonesProcessor& _zp_sel;

	QObject* _img_update_receiver;
};

}

#endif // __PVSCATTERVIEWIMAGESMANAGER_H__
