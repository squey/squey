//! \file PVProgressBox.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <PVProgressBox.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QTimer>

/******************************************************************************
 *
 * PVInspector::PVProgressBox::PVProgressBox
 *
 *****************************************************************************/
PVInspector::PVProgressBox::PVProgressBox(QString msg, QWidget *parent, Qt::WindowFlags flags): QDialog(parent,flags) 
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
	
	widgetCancel = new QWidget(this);
	layoutCancel = new QHBoxLayout();
	widgetCancel->setLayout(layoutCancel);
	_btnCancel = new QPushButton(QString(tr("Cancel")));
	layoutCancel->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
	layoutCancel->addWidget(_btnCancel);
		
	//layout->addItem(layoutCancel);
	layout->addWidget(widgetCancel);
	connect(_btnCancel,SIGNAL(clicked()),this,SLOT(reject()));

	_status = 0;
}

void PVInspector::PVProgressBox::launch_timer_status()
{
	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update_status_Slot()));
	timer->start(10);
}

void PVInspector::PVProgressBox::set_status(int status)
{
	_status = status;
}

void PVInspector::PVProgressBox::update_status_Slot()
{
	progress_bar->setValue(_status);
}

/******************************************************************************
 *
 * PVInspector::PVProgressBox::getProgressBar
 *
 *****************************************************************************/
QProgressBar *PVInspector::PVProgressBox::getProgressBar(){
	return progress_bar;
}

void PVInspector::PVProgressBox::set_enable_cancel(bool enable)
{
	_btnCancel->setEnabled(enable);
}
