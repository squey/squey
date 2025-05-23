/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVWIDGETS_PVAXISCOMBOBOX_H
#define PVWIDGETS_PVAXISCOMBOBOX_H

#include <squey/PVAxesCombination.h>
#include <squey/export.h>

#include <QComboBox>

namespace PVWidgets
{

/**
 * This widget is a combo box to choose an axis among those in a PVAxesCombination.
 */
class PVSQUEY_EXPORT PVAxisComboBox : public QComboBox
{
	Q_OBJECT
  public:
	enum AxesShown {
		OriginalAxes = 0b01,
		CombinationAxes = 0b10,
		BothOriginalCombinationAxes = OriginalAxes | CombinationAxes
	};

	using axes_filter_t = std::function<bool(PVCol, PVCombCol)>;

	explicit PVAxisComboBox(Squey::PVAxesCombination const& axes_comb,
	                        AxesShown shown = AxesShown::OriginalAxes,
	                        axes_filter_t axes_filter = [](PVCol, PVCombCol) { return true; },
	                        QWidget* parent = nullptr);

	void set_current_axis(PVCol axis);
	void set_current_axis(PVCombCol axis);
	PVCol current_axis() const;

	void refresh_axes();

	static constexpr auto MIME_TYPE_PVCOL = "application/vnd.squey.pvcol";

  Q_SIGNALS:
	void current_axis_changed(PVCol axis_col, PVCombCol axis_comb_col);

  protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dropEvent(QDropEvent* event) override;

  private:
	Squey::PVAxesCombination const& _axes_comb;
	AxesShown _axes_shown;
	QPoint _drag_start_position;
	axes_filter_t _axes_filter;
};
} // namespace PVWidgets

#endif
