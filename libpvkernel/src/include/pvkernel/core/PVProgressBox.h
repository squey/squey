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
		PVThreadWatcher* watcher = new PVThreadWatcher();
		connect(watcher, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<Tret, F>, boost::ref(f), boost::ref(ret)));
		return process_worker_thread(watcher, worker, pbox);
	}

	template <typename F>
	static bool progress(F f, PVProgressBox* pbox)
	{
		PVThreadWatcher* watcher = new PVThreadWatcher();
		connect(watcher, SIGNAL(finished()), pbox, SLOT(accept()));
		boost::thread worker(boost::bind(&PVProgressBox::worker_thread<F>, boost::ref(f)));
		return process_worker_thread(watcher, worker, pbox);
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

public slots:
	void update_status_Slot();

private:
	static bool process_worker_thread(PVThreadWatcher* watcher, boost::thread& worker, PVProgressBox* pbox);

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
