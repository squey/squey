/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __PVGUIQT_PVAXESCOMBINATIONEXPORTWIDGET_H__
#define __PVGUIQT_PVAXESCOMBINATIONEXPORTWIDGET_H__

#include <pvguiqt/export.h>
#include <pvkernel/core/PVColumnIndexes.h>
#include <squey/PVAxesCombination.h>

#include <QString>
#include <QStringList>
#include <QWidget>

namespace Squey
{
class PVView;
}

namespace PVGuiQt
{

/**
 * The axes of a view an export writes: all of them, those of its combination, or those of a
 * combination edited for the export, which starts as that of the view.
 */
class PVGUIQT_EXPORT PVAxesCombinationExportWidget : public QWidget
{
	Q_OBJECT

  public:
	enum class Choice { All, Current, Custom };

	/**
	 * Starts with @p choice. @p all_label names the choice of all of the axes. @p view has to
	 * outlive the widget.
	 */
	PVAxesCombinationExportWidget(Squey::PVView const& view,
	                              Choice choice,
	                              QString const& all_label,
	                              QWidget* parent = nullptr);

	Choice choice() const { return _choice; }

	/** The NRAW columns of the axes chosen, in their order. */
	PVCore::PVColumnIndexes column_indexes() const;

	/** The names of the axes chosen, in their order. */
	QStringList names() const;

  Q_SIGNALS:
	/** The choice changed, or the custom combination chosen was edited. */
	void changed();

  private:
	Squey::PVView const& _view;
	Squey::PVAxesCombination _custom_axes_combination;
	Choice _choice;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVAXESCOMBINATIONEXPORTWIDGET_H__
