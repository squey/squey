//! \file PVProgressBox.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVProgressBox.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QTimer>

/******************************************************************************
 *
 * PVCore::PVProgressBox::PVProgressBox
 *
 *****************************************************************************/
PVCore::PVProgressBox::PVProgressBox(QString msg, QWidget *parent, Qt::WindowFlags flags, QString const& format_detail):
	QDialog(parent,flags),
	_format_detail(format_detail)
{
	QVBoxLayout *layout;
	QHBoxLayout *layoutCancel;
	QWidget *widgetCancel;
	
	//set the dialog during the sort
	layout = new QVBoxLayout();
	setLayout(layout);
	
	//message
	message = new QLabel(msg);
	layout->addWidget(message);
	
	//progress bar
	progress_bar = new QProgressBar(this);
	layout->addWidget(progress_bar);
	//by default we don't know the progress
	progress_bar->setMaximum(0);
	progress_bar->setMinimum(0);

	if (!format_detail.isEmpty()) {
		_detail_label = new QLabel();
		layout->addWidget(_detail_label);
	}
	
	widgetCancel = new QWidget(this);
	layoutCancel = new QHBoxLayout();
	widgetCancel->setLayout(layoutCancel);
	_btnCancel = new QPushButton(QString(tr("Cancel")));
	layoutCancel->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
	layoutCancel->addWidget(_btnCancel);
		
	//layout->addItem(layoutCancel);
	layout->addWidget(widgetCancel);
	connect(_btnCancel,SIGNAL(clicked()),this,SLOT(reject()));

	setWindowTitle(msg);

	_status = 0;
}

void PVCore::PVProgressBox::launch_timer_status()
{
	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update_status_Slot()));
	timer->start(10);
}

void PVCore::PVProgressBox::set_status(int status)
{
	_status = status;
}

void PVCore::PVProgressBox::update_status_Slot()
{
	progress_bar->setValue(_status);
	if (!_format_detail.isEmpty()) {
		_detail_label->setText(_format_detail.arg(_status).arg(progress_bar->maximum()));
	}
}

/******************************************************************************
 *
 * PVCore::PVProgressBox::getProgressBar
 *
 *****************************************************************************/
QProgressBar *PVCore::PVProgressBox::getProgressBar(){
	return progress_bar;
}

void PVCore::PVProgressBox::set_enable_cancel(bool enable)
{
	_btnCancel->setEnabled(enable);
}
