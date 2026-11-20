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

#include <QString>
#include <QWidget>

#include <functional>
#include <vector>

class QHBoxLayout;
class QLabel;
class QProgressBar;
class QPushButton;
class QVBoxLayout;

namespace PVGuiQt
{

/**
 * The page of the main window that imports run in: what they import, their progress and what went
 * wrong, rather than in dialogs of their own. Its buttons only tell what the user asks for; what
 * runs the imports acts on it.
 */
class PVGUIQT_EXPORT PVImportPage : public QWidget
{
	Q_OBJECT

  public:
	enum class Severity { information, warning, critical };

	/**
	 * A button of a progress, or of a message, which it dismisses before calling @p on_click.
	 */
	struct Action {
		QString text;
		std::function<void()> on_click;
	};

  public:
	explicit PVImportPage(QWidget* parent = nullptr);

  public:
	void set_title(QString const& title);

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

  Q_SIGNALS:
	void close_requested();

  private:
	QLabel* _title;
	QVBoxLayout* _messages;
	QWidget* _progress;
	QLabel* _task;
	QProgressBar* _bar;
	QLabel* _details;
	QHBoxLayout* _progress_buttons;
	QPushButton* _close;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVIMPORTPAGE_H__
