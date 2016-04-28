/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSCATTERVIEWDATAIMPL_H_
#define PVSCATTERVIEWDATAIMPL_H_

#include <pvparallelview/PVScatterViewDataInterface.h>

#include <pvkernel/core/PVMemory2D.h>

namespace PVParallelView
{

class PVScatterViewDataImpl : public PVScatterViewDataInterface
{
  public:
	void process_bg(ProcessParams const& params, PVScatterViewImage& image,
	                tbb::task_group_context* ctxt = nullptr) const override;
	void process_sel(ProcessParams const& params, PVScatterViewImage& image,
	                 Inendi::PVSelection const& sel,
	                 tbb::task_group_context* ctxt = nullptr) const override;

  private:
	static void process_image(ProcessParams const& params, PVScatterViewImage& image,
	                          Inendi::PVSelection const* sel = nullptr,
	                          tbb::task_group_context* ctxt = nullptr);
};
}

#endif // PVSCATTERVIEWDATAIMPL_H_
