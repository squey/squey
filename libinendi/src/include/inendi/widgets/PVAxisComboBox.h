/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVWIDGETS_PVAXISCOMBOBOX_H
#define PVWIDGETS_PVAXISCOMBOBOX_H

#include <inendi/PVAxesCombination.h>

#include <QComboBox>

namespace PVWidgets
{

/**
 * This widget is a combo box to choose an axis among those in a PVAxesCombination.
 */
class PVAxisComboBox : public QComboBox
{
	Q_OBJECT
  public:
	enum AxesShown {
		OriginalAxes = 0b01,
		CombinationAxes = 0b10,
		BothOriginalCombinationAxes = OriginalAxes | CombinationAxes
	};

	using axes_filter_t = std::function<bool(PVCol, PVCombCol)>;

	explicit PVAxisComboBox(Inendi::PVAxesCombination const& axes_comb,
	                        AxesShown shown = AxesShown::OriginalAxes,
	                        axes_filter_t axes_filter = [](PVCol, PVCombCol) { return true; },
	                        QWidget* parent = nullptr);

	void set_current_axis(PVCol axis);
	void set_current_axis(PVCombCol axis);
	PVCol current_axis() const;

	void refresh_axes();

	static constexpr auto MIME_TYPE_PVCOL = "application/vnd.inendi.pvcol";

  Q_SIGNALS:
	void current_axis_changed(PVCol axis_col, PVCombCol axis_comb_col);

  protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dropEvent(QDropEvent* event) override;

  private:
	Inendi::PVAxesCombination const& _axes_comb;
	AxesShown _axes_shown;
	QPoint _drag_start_position;
	axes_filter_t _axes_filter;
};
} // namespace PVWidgets

#endif