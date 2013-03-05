#ifndef PVPARALLELVIEW_PVHITGRAPHDATAZTRIGHT_H
#define PVPARALLELVIEW_PVHITGRAPHDATAZTRIGHT_H

#include <pvparallelview/PVHitGraphBuffer.h>
#include <pvparallelview/PVHitGraphDataInterface.h>

namespace PVParallelView {

class PVHitGraphDataZTRight: public PVHitGraphDataInterface
{
public:
	void process_all(ProcessParams const& params);
	void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) override;
};

}

#endif
