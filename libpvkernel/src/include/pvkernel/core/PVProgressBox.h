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

#include <pvkernel/core/general.h>

namespace PVCore {
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
