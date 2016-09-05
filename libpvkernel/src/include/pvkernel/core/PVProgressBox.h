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
#include <QMutex>
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
	void set_status(int status);
	void set_enable_cancel(bool cancel);
	void set_extended_status(QString const& str);
	void set_cancel_btn_text(QString const& str);
	void set_cancel2_btn_text(QString const& str);
	void set_detail_label(QString const& detail) { _format_detail = detail; }
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

  public Q_SLOTS:
	void update_status_Slot();

	/**
	 * These function are use to have blocking message in threads.
	 */
  public:
	void critical(QString const& title, QString const& msg)
	{
		std::unique_lock<std::mutex> lk(_blocking_msg);
		Q_EMIT sig_critical(title, msg);
		_cv.wait(lk);
	}

  public Q_SLOTS:
	void critical_slot(QString const& title, QString const& msg)
	{
		{
			std::lock_guard<std::mutex> lk(_blocking_msg);
			QMessageBox::critical(this, title, msg);
		}
		_cv.notify_one();
	}

  Q_SIGNALS:
	void sig_critical(QString const& title, QString const& msg);

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
	int _status;
	QPushButton* _btnCancel;
	QPushButton* _btnCancel2;
	QString _format_detail;
	QString _extended_status;
	QLabel* _detail_label;
	QLabel* _extended_detail_label;
	QMutex _ext_str_mutex;
	volatile CancelState _cancel_state = CancelState::CONTINUE;
	bool _need_confirmation = false;
	std::mutex _blocking_msg; //!< Mutex to have blocking message during thread execution.
	std::condition_variable
	    _cv; //!< Condition variable to sync thread and message during thread execution.
};
}

#endif
