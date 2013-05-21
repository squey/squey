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
	void process_bg(ProcessParams const& params, tbb::task_group_context* ctxt = nullptr) override;
	void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt = nullptr) override;
};

}

#endif // PVSCATTERVIEWDATAIMPL_H_
