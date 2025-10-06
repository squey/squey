//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QApplication>
#include <QStyle>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QMessageBox>

#include <boost/thread.hpp>  // IWYU pragma: keep

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
	widgetCancel->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
	_btnCancel2 = new QPushButton(PVModdedIcon("circle-check"),
	                              QString(tr("")));
	_btnCancel = new QPushButton(PVModdedIcon("circle-x"),
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
	connect(this, &PVProgressBox::set_message_sig, this,
	        &PVProgressBox::set_message_slot, Qt::QueuedConnection);
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
	connect(this, &PVProgressBox::information_sig, this, &PVProgressBox::information_slot,
	        Qt::BlockingQueuedConnection);
	connect(this, &PVProgressBox::finished_sig, this, &PVProgressBox::accept, Qt::QueuedConnection);
	connect(this, &PVProgressBox::canceled_sig, this, &PVProgressBox::reject, Qt::QueuedConnection);

	/* this one must be in blocking mode because the sender must wait for an user interaction
	 */
	connect(this, &PVProgressBox::exec_gui_sig, this, &PVProgressBox::exec_gui_slot,
	        Qt::BlockingQueuedConnection);

	qApp->processEvents();
}

void PVCore::PVProgressBox::cancel()
{
	if (_need_confirmation) {
		QMessageBox confirm(QMessageBox::Question, tr("Confirm"), tr("Are you sure?"),
		                    QMessageBox::Yes | QMessageBox::No, this);
		connect(this, &QDialog::accepted, &confirm, &QDialog::accept);
		if (confirm.exec() == QMessageBox::No) {
			return;
		}
	}
	_btnCancel->setEnabled(false);
	Q_EMIT cancel_asked_sig();
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
	Q_EMIT set_maximum_sig(v);
}

void PVCore::PVProgressBox::set_message(QString const& str)
{
	Q_EMIT set_message_sig(str);
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

void PVCore::PVProgressBox::information(QString const& title, QString const& msg)
{
	Q_EMIT information_sig(title, msg);
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

void PVCore::PVProgressBox::set_message_slot(QString const& str)
{
	message->setText(str);
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

void PVCore::PVProgressBox::information_slot(QString const& title, QString const& msg)
{
	QMessageBox::information(this, title, msg);
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
	PVCore::PVProgressBox::process_t f,
	QString const& name,
	QWidget* parent)
{
	PVProgressBox pbox(name, parent);

	connect(&pbox, &PVProgressBox::cancel_asked_sig, [&](){
		Q_EMIT pbox.canceled_sig();
	});

	boost::thread th([&]() {
		try {
			f(pbox);
		} catch (boost::thread_interrupted) {
		}
		Q_EMIT pbox.finished_sig();
	});

	if (!th.timed_join(boost::posix_time::milliseconds(250))) {
		if (pbox.exec() != QDialog::Accepted) {
			pbox.set_extended_status_slot("Canceling");
			pbox.update();
			th.interrupt();
		}
	}

	th.join();

	return pbox.get_cancel_state();
}
