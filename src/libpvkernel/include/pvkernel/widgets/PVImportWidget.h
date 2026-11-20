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

#ifndef __PVWIDGETS_PVIMPORTWIDGET_H__
#define __PVWIDGETS_PVIMPORTWIDGET_H__

#include <pvkernel/export.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QList>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <utility>
#include <vector>

namespace PVCore
{
class PVTaskProgress;
} // namespace PVCore

namespace PVWidgets
{

/**
 * What an input type asks for an import, as a part of the import page of the main window, between
 * its title and its Import button. Its long tasks, such as reading the files it is given, run out
 * of the GUI thread while the page shows their progress.
 */
class PVKERNEL_EXPORT PVImportWidget : public QWidget
{
	Q_OBJECT

  public:
	enum class Severity { information, warning, critical };
	Q_ENUM(Severity)

	/** The inputs of a source, and their format. */
	using source_t = std::pair<QList<PVRush::PVInputDescription_p>, PVRush::PVFormat>;

	/**
	 * Runs work out of the GUI thread while the page shows its title and the progress work
	 * reports, then done back in the GUI thread, with whether work completed: neither canceled
	 * by the user nor ended by an exception, which the page tells about.
	 */
	using task_runner_t = std::function<void(QString const& title,
	                                         std::function<void(PVCore::PVTaskProgress&)> work,
	                                         std::function<void(bool completed)> done)>;

  public:
	explicit PVImportWidget(QWidget* parent = nullptr) : QWidget(parent) {}

  public:
	/** Whether what the widget holds can be imported. */
	virtual bool is_ready() const = 0;

	/**
	 * The sources to import, once is_ready().
	 *
	 * @throw std::runtime_error with what to tell the user, if they cannot be
	 */
	virtual std::vector<source_t> sources() const = 0;

	/** Starts from the files @p paths, such as those of a command line, for a type of files. */
	virtual void set_paths(QStringList const& /*paths*/) {}
	/**
	 * Whether the widget takes all the room of the page, as a dialog of many panels needs, rather
	 * than a column of the height it asks for.
	 */
	virtual bool fills_page() const { return false; }

	/** How the page runs the long tasks of the widget, which it sets before showing it. */
	void set_task_runner(task_runner_t runner) { _run_task = std::move(runner); }

  Q_SIGNALS:
	void ready_changed();
	/** What the page tells the user, until the next import. */
	void message(PVWidgets::PVImportWidget::Severity severity, QString const& text);

  protected:
	void run_task(QString const& title,
	              std::function<void(PVCore::PVTaskProgress&)> work,
	              std::function<void(bool completed)> done) const
	{
		_run_task(title, std::move(work), std::move(done));
	}

  private:
	task_runner_t _run_task;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVIMPORTWIDGET_H__
