/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef PVPARALLELVIEW_PVSERIESVIEWSELECTIONSQUARE_H
#define PVPARALLELVIEW_PVSERIESVIEWSELECTIONSQUARE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVSelectionRectangle.h>

#include <pvparallelview/PVSeriesView.h>

namespace PVParallelView
{

class PVSeriesViewSelectionRectangle : public PVSelectionRectangle
{
	Q_OBJECT
  public:
	explicit PVSeriesViewSelectionRectangle(QGraphicsScene* scene, PVSeriesView& sv);

	void clear() override;

	void update_position();

  protected:
	void commit(bool use_selection_modifiers) override;

  private:
	PVSeriesView& _sv;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSERIESVIEWSELECTIONSQUARE_H
