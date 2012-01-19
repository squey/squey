//! \file PVProgressBox.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVPROGRESSBOX_H
#define	PVCORE_PVPROGRESSBOX_H

#include <QDialog>
#include <QString>
#include <QLabel>
#include <QProgressBar>
#include <QFuture>
#include <QFutureWatcher>
#include <QtCore>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <tbb/task.h>

#include <pvkernel/core/general.h>

namespace PVCore {

class LibKernelDecl PVThreadWatcher: public QObject
{
	Q_OBJECT
public:
	PVThreadWatcher(boost::thread& thread)
	{
		set_thread(thread);
	}
	PVThreadWatcher() { _thread = NULL; }
	
	void set_thread(boost::thread& thread)
	{
		_thread = &thread;
		boost::thread watcher(boost::bind(&PVThreadWatcher::watch, this));
	}
private:
	void watch()
	{
		_thread->join();
		emit finished();
	}
signals:
	void finished();

private:
	boost::thread* _thread;
};

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

private:
	template <class F>
	static void worker_thread(F f)
	{
		try {
			f();
		}
		catch (boost::thread_interrupted) {
		}
	}

	template <class Tret, class F>
	static void worker_thread(F f, Tret& ret)
	{
		try {
			ret = f();
		}
		catch (boost::thread_interrupted) {
		}
	}

public:
	template <typename Tret, typename F>
	static bool progress(F f, PVProgressBox* pbox, Tret& ret)
	{
		PVThreadWatcher watcher;
		connect(&watcher, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<Tret, F>, boost::ref(f), boost::ref(ret)));
		watcher.set_thread(worker);
		if (pbox->exec() != QDialog::Accepted) {
			worker.interrupt();
			worker.join();
			return false;
		}
		return true;
	}

	template <typename F>
	static bool progress(F f, PVProgressBox* pbox)
	{
		PVThreadWatcher watcher;
		connect(&watcher, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<F>, boost::ref(f)));
		watcher.set_thread(worker);
		if (pbox->exec() != QDialog::Accepted) {
			worker.interrupt();
			worker.join();
			return false;
		}
		return true;
	}

	static bool progress(tbb::task& root_task, PVProgressBox* pbox)
	{
		// This will be the thread that executes the root task
		typedef boost::function<void()> spawn_f;
		PVThreadWatcher watcher;
		connect(&watcher, SIGNAL(finished()), pbox, SLOT(accept()));
		spawn_f f = boost::bind(static_cast<void(*)(tbb::task&)>(&tbb::task::spawn_root_and_wait), boost::ref(root_task));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<spawn_f>, boost::ref(f)));
		watcher.set_thread(worker);
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
	QLabel *message;
	QProgressBar *progress_bar;
	int _status;
	QPushButton *_btnCancel;
	QString _format_detail;
	QLabel* _detail_label;
};

}

#endif
