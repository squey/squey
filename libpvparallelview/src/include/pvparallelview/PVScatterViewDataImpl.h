/**
 * \file PVScatterViewDataImpl.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVSCATTERVIEWDATAIMPL_H_
#define PVSCATTERVIEWDATAIMPL_H_

#include <pvparallelview/PVScatterViewDataInterface.h>

namespace PVParallelView {

class PVScatterViewDataImpl: public PVScatterViewDataInterface
{
public:
	PVScatterViewDataImpl() {};

public:
	void process_bg(ProcessParams const& params) override
	{
		PVZoomedZoneTree::context_t ctxt;

		params.zzt.browse_bci_by_y1_y2(
			ctxt,
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_all().get_hsv_image()
		);

		image_all().convert_image_from_hsv_to_rgb();
	};

	void process_sel(ProcessParams const& params, Picviz::PVSelection const& /*sel*/) override
	{
		PVZoomedZoneTree::context_t ctxt;

		params.zzt.browse_bci_by_y1_y2(
			ctxt,
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_sel().get_hsv_image()
		);

		image_sel().convert_image_from_hsv_to_rgb();
	};
};

}

#endif // PVSCATTERVIEWDATAIMPL_H_
