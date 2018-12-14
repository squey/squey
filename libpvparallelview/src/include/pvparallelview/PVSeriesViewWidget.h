/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef __PVPARALLELVIW_PVSERIESVIEWWIDGET_H__
#define __PVPARALLELVIW_PVSERIESVIEWWIDGET_H__

#include <QWidget>

#include <inendi/PVView.h>

namespace Inendi
{
class PVRangeSubSampler;
}

namespace PVParallelView
{

class PVSeriesViewWidget : public QWidget
{
  public:
	PVSeriesViewWidget(Inendi::PVView* view, PVCombCol axis_comb, QWidget* parent = nullptr);

  private:
	std::unique_ptr<Inendi::PVRangeSubSampler> _sampler;
};
}

#endif // __PVPARALLELVIW_PVSERIESVIEWWIDGET_H__
