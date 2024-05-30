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

#ifndef PVGUIQT_PVAXESCOMBINATIONDIALOG_H
#define PVGUIQT_PVAXESCOMBINATIONDIALOG_H

#include <squey/PVAxesCombination.h>

#include <QDialog>
#include <QDialogButtonBox>

namespace Squey
{
class PVView;
} // namespace Squey

namespace PVGuiQt
{

class PVAxesCombinationWidget;

class PVAxesCombinationDialog : public QDialog
{
	Q_OBJECT

  public:
	explicit PVAxesCombinationDialog(Squey::PVView& view, QWidget* parent = nullptr);

  public:
	void reset_used_axes();

  private:
	Squey::PVView& lib_view() { return _lib_view; }

  protected Q_SLOTS:
	void commit_axes_comb_to_view();
	void box_btn_clicked(QAbstractButton* btn);

  protected:
	PVAxesCombinationWidget* _axes_widget;
	Squey::PVAxesCombination _temp_axes_comb;
	Squey::PVView& _lib_view;

  private:
	QDialogButtonBox* _box_buttons;
};
} // namespace PVGuiQt

#endif
