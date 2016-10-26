/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/PVProgressBox.h>

#include <QApplication>
#include <QStyle>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QMessageBox>

#include <boost/thread.hpp>

/******************************************************************************
 *
 * PVCore::PVProgressBox::PVProgressBox
 *
 *****************************************************************************/
PVCore::PVProgressBox::PVProgressBox(QString msg, QWidget* parent)
    : QDialog(parent), _cancel_state(CancelState::CONTINUE), _need_confirmation(false)
{
	hide();
	QVBoxLayout* layout;
	QHBoxLayout* layoutCancel;
	QWidget* widgetCancel;

	// set the dialog during the sort
	layout = new QVBoxLayout();
	setLayout(layout);

	// message
	message = new QLabel(msg);
	layout->addWidget(message);

	// progress bar
	progress_bar = new QProgressBar(this);
	layout->addWidget(progress_bar);
	// by default we don't know the progress
	progress_bar->setMaximum(0);
	progress_bar->setMinimum(0);
	progress_bar->setValue(0);

	_extended_detail_label = new QLabel();
	_extended_detail_label->setVisible(false);
	layout->addWidget(_extended_detail_label);

	widgetCancel = new QWidget(this);
	layoutCancel = new QHBoxLayout();
	widgetCancel->setLayout(layoutCancel);
	_btnCancel2 = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOkButton),
	                              QString(tr("")));
	_btnCancel = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogCancelButton),
	                             QString(tr("Cancel")));
	_btnCancel2->setVisible(false);
	layoutCancel->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));
	layoutCancel->addWidget(_btnCancel2);
	layoutCancel->addWidget(_btnCancel);

	// layout->addItem(layoutCancel);
	layout->addWidget(widgetCancel);
	connect(_btnCancel, &QPushButton::clicked, [&] {
		_cancel_state = CancelState::CANCEL;
		cancel();
	});
	connect(_btnCancel2, &QPushButton::clicked, [&] {
		_cancel_state = CancelState::CANCEL2;
		cancel();
	});

	qRegisterMetaType<func_t>();

	setWindowTitle(msg);

	/* doing all connection for inter-threads signals/slots emission. As PVProgressBox can be
	 * updated from a thread which is not the main thread (the one owning the GUI), doing queued
	 * signals emission garanties that the slot is called in the receiver thread.
	 */
	connect(this, &PVProgressBox::set_enable_cancel_sig, this,
	        &PVProgressBox::set_enable_cancel_slot, Qt::QueuedConnection);
	connect(this, &PVProgressBox::set_extended_status_sig, this,
	        &PVProgressBox::set_extended_status_slot, Qt::QueuedConnection);
	connect(this, &PVProgressBox::set_value_sig, this, &PVProgressBox::set_value_slot,
	        Qt::QueuedConnection);
	connect(this, &PVProgressBox::set_maximum_sig, this, &PVProgressBox::set_maximum_slot,
	        Qt::QueuedConnection);
	connect(this, &PVProgressBox::set_cancel_btn_text_sig, this,
	        &PVProgressBox::set_cancel_btn_text_slot, Qt::QueuedConnection);
	connect(this, &PVProgressBox::set_cancel2_btn_text_sig, this,
	        &PVProgressBox::set_cancel2_btn_text_slot, Qt::QueuedConnection);
	connect(this, &PVProgressBox::critical_sig, this, &PVProgressBox::critical_slot,
	        Qt::BlockingQueuedConnection);
	connect(this, &PVProgressBox::warning_sig, this, &PVProgressBox::warning_slot,
	        Qt::BlockingQueuedConnection);
	connect(this, &PVProgressBox::finished_sig, this, &PVProgressBox::accept, Qt::QueuedConnection);

	/* this one must be in blocking mode because the sender must wait for an user interaction
	 */
	connect(this, &PVProgressBox::exec_gui_sig, this, &PVProgressBox::exec_gui_slot,
	        Qt::BlockingQueuedConnection);
}

void PVCore::PVProgressBox::cancel()
{
	if (_need_confirmation) {
		QMessageBox confirm(QMessageBox::Question, tr("Confirm"), tr("Are you sure?"),
		                    QMessageBox::Yes | QMessageBox::No, this);
		connect(this, SIGNAL(accepted()), &confirm, SLOT(accept()));
		if (confirm.exec() == QMessageBox::No) {
			return;
		}
	}
	reject();
}

void PVCore::PVProgressBox::set_enable_cancel(bool enable)
{
	Q_EMIT set_enable_cancel_sig(enable);
}

void PVCore::PVProgressBox::set_value(int v)
{
	Q_EMIT set_value_sig(v);
}

void PVCore::PVProgressBox::set_maximum(int v)
{
	Q_EMIT set_value_sig(v);
}

void PVCore::PVProgressBox::set_extended_status(QString const& str)
{
	Q_EMIT set_extended_status_sig(str);
}

void PVCore::PVProgressBox::set_extended_status(std::string const& str)
{
	set_extended_status(QString::fromStdString(str));
}

void PVCore::PVProgressBox::set_extended_status(const char* str)
{
	set_extended_status(QString(str));
}

void PVCore::PVProgressBox::set_cancel_btn_text(QString const& str)
{
	Q_EMIT set_cancel_btn_text_sig(str);
}

void PVCore::PVProgressBox::set_cancel2_btn_text(QString const& str)
{
	Q_EMIT set_cancel2_btn_text_sig(str);
}

void PVCore::PVProgressBox::critical(QString const& title, QString const& msg)
{
	Q_EMIT critical_sig(title, msg);
}

void PVCore::PVProgressBox::warning(QString const& title, QString const& msg)
{
	Q_EMIT warning_sig(title, msg);
}

void PVCore::PVProgressBox::exec_gui(PVCore::PVProgressBox::func_t f)
{
	Q_EMIT exec_gui_sig(f);
}

void PVCore::PVProgressBox::set_value_slot(int v)
{
	progress_bar->setValue(v);
}

void PVCore::PVProgressBox::set_maximum_slot(int v)
{
	progress_bar->setMaximum(v);
}

void PVCore::PVProgressBox::set_extended_status_slot(QString const& str)
{
	_extended_detail_label->setVisible(true);
	_extended_detail_label->setText(str);
}

void PVCore::PVProgressBox::set_cancel_btn_text_slot(QString const& str)
{
	_btnCancel->setText(str);
}

void PVCore::PVProgressBox::set_cancel2_btn_text_slot(QString const& str)
{
	_btnCancel2->setVisible(true);
	_btnCancel2->setText(str);
}

void PVCore::PVProgressBox::critical_slot(QString const& title, QString const& msg)
{
	QMessageBox::critical(this, title, msg);
}

void PVCore::PVProgressBox::warning_slot(QString const& title, QString const& msg)
{
	QMessageBox::warning(this, title, msg);
}

void PVCore::PVProgressBox::set_enable_cancel_slot(bool enable)
{
	_btnCancel->setEnabled(enable);
}

void PVCore::PVProgressBox::exec_gui_slot(PVCore::PVProgressBox::func_t f)
{
	f();
}

PVCore::PVProgressBox::CancelState PVCore::PVProgressBox::progress(
    PVCore::PVProgressBox::process_t f, QString const& name, QWidget* parent)
{
	PVProgressBox pbox(name, parent);

	boost::thread th([&]() { pbox.process(f); });

	if (!th.timed_join(boost::posix_time::milliseconds(250))) {
		if (pbox.exec() != QDialog::Accepted) {
			pbox.set_extended_status_slot("Cancelling");
			pbox.update();
			th.interrupt();
		}
	}

	th.join();
	return pbox.get_cancel_state();
}

void PVCore::PVProgressBox::process(process_t f)
{
	try {
		f(*this);
	} catch (boost::thread_interrupted) {
	}

	Q_EMIT finished_sig();
}
