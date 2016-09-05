/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVProgressBox.h>

#include <QApplication>
#include <QStyle>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QMessageBox>

/******************************************************************************
 *
 * PVCore::PVProgressBox::PVProgressBox
 *
 *****************************************************************************/
PVCore::PVProgressBox::PVProgressBox(QString msg, QWidget* parent) : QDialog(parent)
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

	connect(this, SIGNAL(sig_critical(QString const&, QString const&)), this,
	        SLOT(critical_slot(QString const&, QString const&)));

	setWindowTitle(msg);
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

void PVCore::PVProgressBox::set_extended_status(QString const& str)
{
	_extended_detail_label->setVisible(true);
	_extended_detail_label->setText(str);
}

void PVCore::PVProgressBox::set_cancel_btn_text(QString const& str)
{
	_btnCancel->setText(str);
}

void PVCore::PVProgressBox::set_cancel2_btn_text(QString const& str)
{
	_btnCancel2->setVisible(true);
	_btnCancel2->setText(str);
}

/******************************************************************************
 *
 * PVCore::PVProgressBox::getProgressBar
 *
 *****************************************************************************/
QProgressBar* PVCore::PVProgressBox::getProgressBar()
{
	return progress_bar;
}

void PVCore::PVProgressBox::set_enable_cancel(bool enable)
{
	_btnCancel->setEnabled(enable);
}

bool PVCore::PVProgressBox::process_worker_thread(__impl::ThreadEndSignal* watcher,
                                                  boost::thread& worker,
                                                  PVProgressBox* pbox)
{
	// watcher->set_thread(worker);
	// Show the window only if the work takes more than 50ms (avoid window flashing)
	if (!worker.timed_join(boost::posix_time::milliseconds(250))) {
		if (pbox->exec() != QDialog::Accepted) {
			worker.interrupt();
			worker.join();
			return false;
		}
	} else {
		disconnect(watcher, SIGNAL(finished()), pbox, SLOT(accept()));
	}
	watcher->deleteLater();
	return true;
}

bool PVCore::PVProgressBox::process_worker_thread(__impl::ThreadEndSignal* watcher,
                                                  boost::thread& worker,
                                                  PVProgressBox* pbox,
                                                  tbb::task_group_context& ctxt)
{
	// watcher->set_thread(worker);
	// Show the window only if the work takes more than 50ms (avoid window flashing)
	if (!worker.timed_join(boost::posix_time::milliseconds(250))) {
		if (pbox->exec() != QDialog::Accepted) {
			ctxt.cancel_group_execution();
			worker.join();
			return false;
		}
	} else {
		disconnect(watcher, SIGNAL(finished()), pbox, SLOT(accept()));
	}
	watcher->deleteLater();
	return true;
}
