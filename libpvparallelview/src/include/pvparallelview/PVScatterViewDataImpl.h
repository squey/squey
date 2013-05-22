/**
 * \file PVScatterViewDataImpl.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVSCATTERVIEWDATAIMPL_H_
#define PVSCATTERVIEWDATAIMPL_H_

#include <pvparallelview/PVScatterViewDataInterface.h>

#include <pvkernel/core/PVMemory2D.h>

namespace PVParallelView {

class PVScatterViewDataImpl: public PVScatterViewDataInterface
{
public:
	void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) override
	{
		process_image(params, image_sel(), &sel);
	}

	void process_bg(ProcessParams const& params) override
	{
		process_image(params, image_all());
	}

private:
	void process_image(ProcessParams const& params, PVScatterViewImage& image, Picviz::PVSelection const* sel = nullptr);
};

}

#endif // PVSCATTERVIEWDATAIMPL_H_
