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

#ifndef PVCORE_PVPROGRESSBOX_H
#define PVCORE_PVPROGRESSBOX_H

#include <QDialog>
#include <QString>

#include <functional>
#include <atomic>

class QLabel;
class QPushButton;
class QProgressBar;

namespace PVCore
{

class PVProgressBox : public QDialog
{
	Q_OBJECT

  public:
	enum class CancelState { CONTINUE, CANCEL, CANCEL2 };

	typedef std::function<void(PVProgressBox&)> process_t;
	typedef std::function<void(void)> func_t;

  protected:
	PVProgressBox(QString msg, QWidget* parent);

  public:
	void set_value(int v);
	void set_maximum(int v);

	void set_message(QString const& str);

	void set_extended_status(QString const& str);
	void set_extended_status(std::string const& str);
	void set_extended_status(const char* str);

	void set_enable_cancel(bool cancel);

	void set_cancel_btn_text(QString const& str);
	void set_cancel2_btn_text(QString const& str);

	void critical(QString const& title, QString const& msg);
	void warning(QString const& title, QString const& msg);
	void information(QString const& title, QString const& msg);

	void exec_gui(std::function<void(void)> f);

	CancelState get_cancel_state() { return _cancel_state; }

	void set_canceled() { _cancel_state = CancelState::CANCEL; }
	void set_confirmation(bool confirm) { _need_confirmation = confirm; }

  Q_SIGNALS:
	void set_value_sig(int v);
	void set_maximum_sig(int v);

	void set_message_sig(QString const& str);

	void set_extended_status_sig(QString const& str);

	void set_enable_cancel_sig(bool cancel);

	void set_cancel_btn_text_sig(QString const& str);
	void set_cancel2_btn_text_sig(QString const& str);

	void critical_sig(QString const& title, QString const& msg);
	void warning_sig(QString const& title, QString const& msg);
	void information_sig(QString const& title, QString const& msg);

	void exec_gui_sig(std::function<void(void)> f);

	void cancel_asked_sig();
	void canceled_sig();
	void finished_sig();

  public Q_SLOTS:
	void set_value_slot(int v);
	void set_maximum_slot(int v);

	void set_message_slot(QString const& str);

	void set_extended_status_slot(QString const& str);

	void set_enable_cancel_slot(bool cancel);

	void set_cancel_btn_text_slot(QString const& str);
	void set_cancel2_btn_text_slot(QString const& str);

	void critical_slot(QString const& title, QString const& msg);
	void warning_slot(QString const& title, QString const& msg);
	void information_slot(QString const& title, QString const& msg);

	void exec_gui_slot(std::function<void(void)> f);

  public:
	static CancelState progress(process_t f, QString const& name, QWidget* parent);

  protected:
	void cancel();

  protected:
	QLabel* message;
	QProgressBar* progress_bar;
	QPushButton* _btnCancel;
	QPushButton* _btnCancel2;
	QLabel* _extended_detail_label;
	std::atomic<CancelState> _cancel_state;
	std::atomic<bool> _need_confirmation;
	std::atomic<bool> _cancel_asked;
};
} // namespace PVCore

Q_DECLARE_METATYPE(std::function<void()>);

#endif
