/**
 * \file PVProgressBox.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVPROGRESSBOX_H
#define	PVCORE_PVPROGRESSBOX_H

#include <QDialog>
#include <QString>
#include <QLabel>
#include <QProgressBar>
#include <QFuture>
#include <QFutureWatcher>
#include <QObject>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <tbb/task.h>

#include <pvkernel/core/general.h>

namespace PVCore {

namespace __impl {

class ThreadEndSignal: public QObject
{
	Q_OBJECT
public:
	void emit_finished() { emit finished(); }
signals:
	void finished();
};

}

class LibKernelDecl PVProgressBox: public QDialog
{
	Q_OBJECT
public:
	PVProgressBox (QString msg, QWidget * parent = 0, Qt::WindowFlags f = 0, QString const& format_detail = QString());
	/**
	* Return the progress bar. It possible to modify Min, Max and progress.
	*/
	QProgressBar *getProgressBar();
	void launch_timer_status();
	void set_status(int status);
	void set_enable_cancel(bool cancel);
	void set_extended_status(QString const& str);

private:
	template <class F>
	static void worker_thread(F f, __impl::ThreadEndSignal* s)
	{
		try {
			f();
		}
		catch (boost::thread_interrupted) {
		}
		s->emit_finished();
	}

	template <class Tret, class F>
	static void worker_thread(F f, Tret& ret, __impl::ThreadEndSignal* s)
	{
		try {
			ret = f();
		}
		catch (boost::thread_interrupted) {
		}
		s->emit_finished();
	}

public:
	template <typename Tret, typename F>
	static bool progress(F f, PVProgressBox* pbox, Tret& ret)
	{
		//PVThreadWatcher* watcher = new PVThreadWatcher();
		__impl::ThreadEndSignal* end_s = new __impl::ThreadEndSignal();
		connect(end_s, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<Tret, F>, boost::ref(f), boost::ref(ret), end_s));
		return process_worker_thread(end_s, worker, pbox);
	}

	template <typename F>
	static bool progress(F f, PVProgressBox* pbox)
	{
		//PVThreadWatcher* watcher = new PVThreadWatcher();
		__impl::ThreadEndSignal* end_s = new __impl::ThreadEndSignal();
		connect(end_s, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<F>, boost::ref(f), end_s));
		return process_worker_thread(end_s, worker, pbox);
	}

	static bool progress(tbb::task& root_task, PVProgressBox* pbox)
	{
		// This will be the thread that executes the root task
		typedef boost::function<void()> spawn_f;
		__impl::ThreadEndSignal* end_s = new __impl::ThreadEndSignal();
		connect(end_s, SIGNAL(finished()), pbox, SLOT(accept()));
		spawn_f f = boost::bind(static_cast<void(*)(tbb::task&)>(&tbb::task::spawn_root_and_wait), boost::ref(root_task));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<spawn_f>, boost::ref(f), end_s));
		if (pbox->exec() != QDialog::Accepted) {
			root_task.cancel_group_execution();
			worker.join();
			return false;
		}
		return true;
	}

	template <typename Tret, typename F>
	static bool progress(F f, QString const& text, Tret& ret, QWidget* parent = NULL)
	{
		PVProgressBox* pbox = new PVProgressBox(text, parent);
		return progress(f, pbox, ret);
	}

	template <typename F>
	static bool progress(F f, QString const& text, QWidget* parent = NULL)
	{
		PVProgressBox* pbox = new PVProgressBox(text, parent);
		return progress(f, pbox);
	}

	static bool progress(tbb::task& root, QString const& text, QWidget* parent = NULL)
	{
		PVProgressBox* pbox = new PVProgressBox(text, parent);
		return progress(root, pbox);
	}

public slots:
	void update_status_Slot();

private:
	static bool process_worker_thread(__impl::ThreadEndSignal* watcher, boost::thread& worker, PVProgressBox* pbox);

private:
	QLabel *message;
	QProgressBar *progress_bar;
	int _status;
	QPushButton *_btnCancel;
	QString _format_detail;
	QString _extended_status;
	QLabel* _detail_label;
	QLabel* _extended_detail_label;
	QMutex _ext_str_mutex;
};

}

#endif
