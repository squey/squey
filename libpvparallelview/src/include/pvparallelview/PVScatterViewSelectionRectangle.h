/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSCATTERVIEWSELECTIONSQUARE_H
#define PVPARALLELVIEW_PVSCATTERVIEWSELECTIONSQUARE_H

#include <pvparallelview/PVSelectionRectangle.h>

#include <pvbase/types.h>

namespace PVParallelView
{

class PVScatterView;

class PVScatterViewSelectionRectangle : public PVSelectionRectangle
{
  public:
	PVScatterViewSelectionRectangle(PVScatterView* sv);

  public:
	void set_plotteds(const uint32_t* y1_plotted, const uint32_t* y2_plotted, const PVRow nrows);

  protected:
	void commit(bool use_selection_modifiers) override;

	Inendi::PVView& lib_view() override;

  private:
	const uint32_t* _y1_plotted;
	const uint32_t* _y2_plotted;
	PVRow _nrows;
	PVScatterView* _sv;
};
}

#endif // PVPARALLELVIEW_PVSCATTERVIEWSELECTIONSQUARE_H
