/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVPROGRESSBOX_H
#define PVCORE_PVPROGRESSBOX_H

#include <QDialog>
#include <QLabel>
#include <QMessageBox>
#include <QObject>
#include <QProgressBar>
#include <QString>

#include <boost/thread.hpp>

#include <tbb/task.h>

#include <condition_variable>

namespace PVCore
{

namespace __impl
{

class ThreadEndSignal : public QObject
{
	Q_OBJECT
  public:
	void emit_finished() { Q_EMIT finished(); }
  Q_SIGNALS:
	void finished();
};
}

class PVProgressBox : public QDialog
{
	Q_OBJECT

  public:
	enum class CancelState { CONTINUE, CANCEL, CANCEL2 };

  private:
	PVProgressBox(QString msg, QWidget* parent);

  public:
	/**
	* Return the progress bar. It possible to modify Min, Max and progress.
	*/
	QProgressBar* getProgressBar();
	void set_enable_cancel(bool cancel);
	void set_extended_status(QString const& str);
	void set_extended_status(std::string const& str)
	{
		set_extended_status(QString::fromStdString(str));
	}
	void set_extended_status(const char* str) { set_extended_status(std::string(str)); }
	void set_cancel_btn_text(QString const& str);
	void set_cancel2_btn_text(QString const& str);
	CancelState get_cancel_state() { return _cancel_state; }
	void set_confirmation(bool confirm) { _need_confirmation = confirm; }

  private:
	template <class F>
	static void worker_thread(F f, __impl::ThreadEndSignal* s, PVProgressBox& pbox)
	{
		try {
			f(pbox);
		} catch (boost::thread_interrupted) {
		}
		s->emit_finished();
	}

  public:
	template <typename F>
	static CancelState progress(F f, QString const& name, QWidget* parent)
	{
		PVProgressBox pbox(name, parent);
		__impl::ThreadEndSignal* end_s = new __impl::ThreadEndSignal();
		connect(end_s, SIGNAL(finished()), &pbox, SLOT(accept()));
		boost::thread worker([&]() { worker_thread<F>(f, end_s, pbox); });
		process_worker_thread(end_s, worker, &pbox);
		return pbox.get_cancel_state();
	}

	template <typename F>
	static CancelState
	progress(F f, tbb::task_group_context& ctxt, QString const& name, QWidget* parent)
	{
		PVProgressBox pbox(name, parent);
		__impl::ThreadEndSignal* end_s = new __impl::ThreadEndSignal();
		connect(end_s, SIGNAL(finished()), &pbox, SLOT(accept()));
		boost::thread worker([&]() { worker_thread<F>(f, end_s, pbox); });
		process_worker_thread(end_s, worker, &pbox, ctxt);
		return pbox.get_cancel_state();
	}

  public:
	void exec_gui(std::function<void()> const& f)
	{
		std::unique_lock<std::mutex> lk(_blocking_msg);
		Q_EMIT sig_exec_gui(f);
		_cv.wait(lk);
	}

	void critical(QString const& title, QString const& msg)
	{
		exec_gui([&]() { QMessageBox::critical(this, title, msg); });
	}
	void warning(QString const& title, QString const& msg)
	{
		exec_gui([&]() { QMessageBox::warning(this, title, msg); });
	}

  public Q_SLOTS:
	void exec_gui_slot(std::function<void()> f)
	{
		{
			std::lock_guard<std::mutex> lk(_blocking_msg);
			f();
		}
		_cv.notify_one();
	}

  Q_SIGNALS:
	void sig_exec_gui(std::function<void()> f);

  private:
	static bool process_worker_thread(__impl::ThreadEndSignal* watcher,
	                                  boost::thread& worker,
	                                  PVProgressBox* pbox);
	static bool process_worker_thread(__impl::ThreadEndSignal* watcher,
	                                  boost::thread& worker,
	                                  PVProgressBox* pbox,
	                                  tbb::task_group_context& ctxt);

  private:
	void cancel();

  private:
	QLabel* message;
	QProgressBar* progress_bar;
	QPushButton* _btnCancel;
	QPushButton* _btnCancel2;
	QLabel* _extended_detail_label;
	volatile CancelState _cancel_state = CancelState::CONTINUE;
	bool _need_confirmation = false;
	std::mutex _blocking_msg; //!< Mutex to have blocking message during thread execution.
	std::condition_variable
	    _cv; //!< Condition variable to sync thread and message during thread execution.
};
}

Q_DECLARE_METATYPE(std::function<void()>);

#endif
