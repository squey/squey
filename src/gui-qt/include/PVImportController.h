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

#ifndef __APP_PVIMPORTCONTROLLER_H__
#define __APP_PVIMPORTCONTROLLER_H__

#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>

#include <QObject>
#include <QPointer>
#include <QString>
#include <QTimer>

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

class QShortcut;
class QThread;

namespace Squey
{
class PVScene;
class PVSource;
} // namespace Squey

namespace PVGuiQt
{
class PVImportPage;
class PVProjectsTabWidget;
} // namespace PVGuiQt

namespace PVWidgets
{
class PVImportWidget;
} // namespace PVWidgets

namespace App
{

class PVMainWindow;
class PVMemoryConsumptionForecast;

/**
 * Imports sources one after the other in the import page of the main window: the extraction of
 * each, then its mapping, its scaling and its view, with their progress and what went wrong on
 * the page rather than in dialogs. The rest of the main window is disabled meanwhile, as a modal
 * dialog had it.
 *
 * The page also shows what an input type asks for an import, its content, until the user imports
 * what it holds, into the data collection they choose, or cancels it.
 */
class PVImportController : public QObject
{
	Q_OBJECT

  public:
	explicit PVImportController(PVMainWindow& window);
	~PVImportController() override;

  public:
	/**
	 * Imports @p sources once those given before are, and the task of the content, if one runs,
	 * each into the current data collection, or into a new one when there is none.
	 */
	void import(std::vector<PVRush::PVSourceDescription> sources);

	/**
	 * Shows @p content, which asks for an import of @p type, in the import page, which owns it.
	 *
	 * @return false if the page runs something else, which it then tells, and @p content is
	 * deleted
	 */
	bool edit(PVRush::PVInputType_p type, PVWidgets::PVImportWidget* content);

	/** Whether sources are imported, or a task of the content runs. */
	bool busy() const;

  Q_SIGNALS:
	/** Once nothing is imported anymore. */
	void idle();

  private:
	enum class State { idle, extracting, processing };

	PVGuiQt::PVProjectsTabWidget& projects() const;
	PVGuiQt::PVImportPage& page() const;

	void import_content();
	void cancel_content();
	/** Starts the imports given while the task of the content ran. */
	void task_finished();

	void begin();
	/** Starts the import of the next source, once the event loop got its turn. */
	void next();
	void start_next();
	void poll();
	void check_memory();
	void stop_extraction();
	void ask_discard();
	/**
	 * Pauses the extraction while @p text asks the user to choose between @p actions, each a text
	 * and what clicking it does.
	 */
	void ask(QString const& text, std::vector<std::pair<QString, std::function<void()>>> actions);
	void extraction_ended();
	void process();
	void processed();
	/** Tells why the source cannot be imported, and drops it. */
	void fail(QString const& title, QString const& text);
	void drop_source();
	void end();
	void set_window_enabled(bool enabled);

  private:
	PVMainWindow& _window;
	std::deque<PVRush::PVSourceDescription> _queue;
	State _state = State::idle;
	bool _busy = false;

	PVRush::PVInputType_p _content_type;
	/** The data collections the content can be imported into, as the page lists them. */
	std::vector<Squey::PVScene*> _collections;
	/** Whether the imports are those of the content, and whether one of them was imported. */
	bool _content_import = false;
	bool _imported = false;

	/** Of the source being imported, as the messages about it name it. */
	QString _name;
	Squey::PVSource* _source = nullptr;
	/** The data collection made for the source, to remove if the source cannot be made. */
	Squey::PVScene* _new_scene = nullptr;
	PVRush::PVControllerJob_p _job;
	QTimer _poll_timer;
	std::unique_ptr<PVMemoryConsumptionForecast> _memory_forecast;
	bool _monitor_memory = true;
	std::chrono::steady_clock::time_point _last_memory_check;
	bool _discarding = false;
	/** The question that pauses the extraction, if any. */
	QPointer<QWidget> _question;

	QThread* _worker = nullptr;
	/** Of the processing that _worker runs, from 0 to 3. */
	std::atomic<int> _processing_step = 0;
	/** Why the processing failed, set by _worker. */
	QString _processing_error;

	/** The shortcuts of the main window, disabled while it imports. */
	std::vector<QPointer<QShortcut>> _disabled_shortcuts;
};

} // namespace App

#endif // __APP_PVIMPORTCONTROLLER_H__
