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

#ifndef __PVWIDGETS_PVDIALOGIMPORTWIDGET_H__
#define __PVWIDGETS_PVDIALOGIMPORTWIDGET_H__

#include <pvkernel/widgets/PVImportWidget.h>

#include <QList>
#include <QPointer>
#include <QStringList>

#include <functional>
#include <vector>

class QAbstractButton;
class QDialog;

namespace PVWidgets
{

/**
 * The dialog in which an input type asks for an import, shown in the import page rather than in a
 * window of its own. The buttons of the dialog hide, as the page has its own: Import clicks the
 * button that accepted the dialog, which runs the checks of the dialog as it did. Enter, which
 * clicked the default button of the dialog, goes on to the page, which imports then.
 */
class PVDialogImportWidget : public PVImportWidget
{
	Q_OBJECT

  public:
	struct Parts {
		/** The button that accepts the dialog, ready while enabled. */
		QAbstractButton* accept;
		/** The buttons of the dialog, which hide. */
		QList<QWidget*> buttons;
		/** What the accepted dialog asks to import. */
		std::function<std::vector<source_t>()> sources;
		/** Adds files to those the dialog imports, if it takes any. */
		std::function<void(QStringList const&)> add_paths = {};
		bool fills_page = false;
	};

  public:
	/** Owns @p dialog, which it rejects as it goes, as closing its window did. */
	PVDialogImportWidget(QDialog* dialog, Parts parts, QWidget* parent = nullptr);
	~PVDialogImportWidget() override;

  public:
	bool is_ready() const override;
	/** Nothing if the dialog refused to be accepted: it tells why itself. */
	std::vector<source_t> sources() const override;
	void set_paths(QStringList const& paths) override { add_paths(paths); }
	bool takes_files() const override { return bool(_parts.add_paths); }
	void add_paths(QStringList const& paths) override;
	bool fills_page() const override { return _parts.fills_page; }

  protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

  private:
	QPointer<QDialog> _dialog;
	Parts _parts;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVDIALOGIMPORTWIDGET_H__
