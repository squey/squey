/**
 * \file PVScatterViewDataImpl.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVSCATTERVIEWDATAIMPL_H_
#define PVSCATTERVIEWDATAIMPL_H_

#include <pvparallelview/PVScatterViewDataInterface.h>
#include <pvparallelview/PVZoomedZoneTree.h>

namespace PVParallelView {

class PVScatterViewDataImpl: public PVScatterViewDataInterface
{
public:
	PVScatterViewDataImpl() {};

public:
	void process_bg(ProcessParams const& params) override
	{
		params.zzt.browse_bci_by_y1_y2(
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_bg().get_hsv_image()
		);

		image_bg().convert_image_from_hsv_to_rgb();
	};

	void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) override
	{
		params.zzt.browse_bci_by_y1_y2_sel(
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_sel().get_hsv_image(),
			sel
		);

		image_sel().convert_image_from_hsv_to_rgb();
	};
};

}

#endif // PVSCATTERVIEWDATAIMPL_H_
