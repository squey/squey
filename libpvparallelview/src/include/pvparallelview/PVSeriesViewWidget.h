/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
#define __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__

#include <QWidget>

#include <inendi/PVView.h>

#include <pvkernel/widgets/PVHelpWidget.h>

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

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void enterEvent(QEvent*) override;
	void leaveEvent(QEvent*) override;

  private:
	std::unique_ptr<Inendi::PVRangeSubSampler> _sampler;

	struct Disconnector : public sigc::connection {
		using sigc::connection::connection;
		using sigc::connection::operator=;
		Disconnector(Disconnector&&) = delete;
		~Disconnector() { disconnect(); }
	};

	Disconnector _plotting_change_connection;
	Disconnector _selection_change_connection;

	PVWidgets::PVHelpWidget _help_widget;
};

struct SerieListItemData {
	PVCol col;
	QColor color;
};
}

Q_DECLARE_METATYPE(PVParallelView::SerieListItemData)

#endif // __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
