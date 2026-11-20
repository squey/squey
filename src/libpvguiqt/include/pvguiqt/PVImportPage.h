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

#ifndef __PVGUIQT_PVIMPORTPAGE_H__
#define __PVGUIQT_PVIMPORTPAGE_H__

#include <pvguiqt/export.h>
#include <pvkernel/widgets/PVImportWidget.h>

#include <QString>
#include <QStringList>
#include <QTimer>
#include <QWidget>

#include <functional>
#include <memory>
#include <vector>

namespace App
{
class PVFormatBuilderWidget;
} // namespace App

class QComboBox;
class QHBoxLayout;
class QLabel;
class QProgressBar;
class QPushButton;
class QThread;
class QVBoxLayout;

namespace PVCore
{
class PVTaskProgress;
} // namespace PVCore

namespace PVGuiQt
{

/**
 * The page of the main window that imports run in: what they import, their progress and what went
 * wrong, rather than in dialogs of their own. It shows what an input type asks for an import, the
 * content, with the same buttons for every type. Its buttons only tell what the user asks for;
 * what runs the imports acts on it.
 */
class PVGUIQT_EXPORT PVImportPage : public QWidget
{
	Q_OBJECT

  public:
	using Severity = PVWidgets::PVImportWidget::Severity;

	/**
	 * A button of a progress, or of a message, which it dismisses before calling @p on_click.
	 */
	struct Action {
		QString text;
		std::function<void()> on_click;
	};

  public:
	explicit PVImportPage(QWidget* parent = nullptr);
	~PVImportPage() override;

  public:
	void set_title(QString const& title);

	/**
	 * Shows @p content under @p title, then a choice among the data collections @p collections,
	 * @p current chosen first, and a new one, and Cancel and Import buttons. The page owns
	 * @p content, and runs its tasks, until clear_content().
	 */
	void set_content(PVWidgets::PVImportWidget* content,
	                 QString const& title,
	                 QStringList const& collections,
	                 int current);
	void clear_content();
	PVWidgets::PVImportWidget* content() const { return _content; }
	/** The index of the chosen one of the collections of set_content(), or -1 for a new one. */
	int chosen_collection() const;
	/** Whether the content and its buttons show, rather than the progress of its import alone. */
	void set_content_shown(bool shown);

	/**
	 * Shows the format builder in place of the content, to make a format from @p format, an empty
	 * one for none, with a preview of @p input of @p type, until the user uses it, which the
	 * content then gets through set_edited_format(), or cancels.
	 */
	void edit_format(QDomDocument const& format,
	                 PVRush::PVInputDescription_p const& input,
	                 std::shared_ptr<PVRush::PVInputType> const& type);
	bool editing_format() const { return _builder != nullptr; }

	/** Runs a task of the content, as PVWidgets::PVImportWidget::task_runner_t tells. */
	void run_task(QString const& title,
	              std::function<void(PVCore::PVTaskProgress&)> work,
	              std::function<void(bool completed)> done);
	bool task_running() const { return _task_thread != nullptr; }

	/** Shows the progress of @p task, whose end is unknown until set_progress() gives it. */
	void start_progress(QString const& task, std::vector<Action> buttons = {});
	/** @p maximum is 0 while the end of the task is unknown. */
	void set_progress(qint64 value, qint64 maximum);
	void set_progress_details(QString const& details);
	void set_progress_buttons_enabled(bool enabled);
	void end_progress();

	/**
	 * Adds a message after those shown, until remove_message(), clear_messages() or one of its
	 * actions.
	 *
	 * @return the message, which the page owns
	 */
	QWidget* add_message(Severity severity, QString const& text, std::vector<Action> actions = {});
	void remove_message(QWidget* message);
	void clear_messages();
	bool has_messages() const;

	/** Whether the Close button shows, for the user to leave the page once nothing runs. */
	void set_closable(bool closable);
	void set_back_enabled(bool enabled);

	/** Asks the task of the content to stop, if one runs: task_finished() tells once it did. */
	void cancel_task();

  Q_SIGNALS:
	/** For the page shown before, while this one keeps what it shows. */
	void back_requested();
	/** From the Close or the Cancel button: what the page shows is to drop. */
	void close_requested();
	void import_requested();
	/** Once a task of the content ended, and its done function returned. */
	void task_finished();

  private:
	void update_buttons();
	/** Whether the content takes all the room of the page, or a column. */
	void fill_page(bool fill);
	void end_format_edit();

  private:
	QPushButton* _back;
	QLabel* _title;
	QWidget* _body;
	QVBoxLayout* _body_layout;
	QVBoxLayout* _messages;
	QWidget* _progress;
	QLabel* _task;
	QProgressBar* _bar;
	QLabel* _details;
	QHBoxLayout* _progress_buttons;
	QVBoxLayout* _content_layout;
	PVWidgets::PVImportWidget* _content = nullptr;
	QWidget* _collection_row;
	QComboBox* _collection;
	QPushButton* _close;
	QPushButton* _cancel;
	QPushButton* _import;
	/** While the user makes the format of the content. */
	App::PVFormatBuilderWidget* _builder = nullptr;
	QPushButton* _format_cancel;
	QPushButton* _format_use;

	/** That of the task of the content, if one runs. */
	QThread* _task_thread = nullptr;
	std::shared_ptr<PVCore::PVTaskProgress> _task_progress;
	QTimer _task_timer;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVIMPORTPAGE_H__
