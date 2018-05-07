/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVCRASHREPORTER_H__
#define __PVGUIQT_PVCRASHREPORTER_H__

#include <pvkernel/core/PVCrashReportSender.h>
#include <pvbase/general.h>

#include <QDialog>
#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QStyle>
#include <QApplication>
#include <QIcon>
#include <QPixmap>
#include <QMessageBox>

namespace PVGuiQt
{

class PVCrashReporterDialog : public QDialog
{
  public:
	PVCrashReporterDialog(const std::string& minidump_path) : _minidump_path(minidump_path)
	{
		setWindowTitle("Crash Reporter");

		QPushButton* send_report = new QPushButton("Send crash report");
		QPushButton* cancel = new QPushButton("Don't send");

		QDialogButtonBox* button_box = new QDialogButtonBox();
		button_box->addButton(send_report, QDialogButtonBox::AcceptRole);
		button_box->addButton(cancel, QDialogButtonBox::RejectRole);

		connect(button_box, &QDialogButtonBox::accepted, this, &PVCrashReporterDialog::send_crash);
		connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

		QVBoxLayout* main_layout = new QVBoxLayout(this);

		QLabel* text = new QLabel(
		    "<b>The program has crashed and we apologize for the inconvenience.</b><br/><br/>"
		    "Please send us the crash report so we can diagnose and fix the problem.<br/>"
		    "We will treat this report as confidential and only to improve this "
		    "product.<br/><br/>");
		QHBoxLayout* text_layout = new QHBoxLayout();
		QLabel* icon_label = new QLabel();
		QPixmap pixmap = QApplication::style()
		                     ->standardIcon(QStyle::SP_MessageBoxCritical)
		                     .pixmap(QSize(32, 32));
		icon_label->setPixmap(pixmap);
		text_layout->addWidget(icon_label);
		text_layout->addWidget(text);

		main_layout->addLayout(text_layout);

		main_layout->addWidget(button_box);
		setLayout(main_layout);
	}

  private:
	void send_crash()
	{
		std::string locking_code = PVCore::PVLicenseActivator::get_locking_code();
		bool ret = PVCore::PVCrashReportSender::send(_minidump_path, INENDI_CURRENT_VERSION_STR,
		                                             locking_code);
		if (not ret) {
			QMessageBox::critical(
			    this, "Error sending crash report",
			    "Error when sending crash report.<br>Please, check your Internet connection.");
		} else {
			QMessageBox::information(this, "Crash report sent with success",
			                         "Your crash report was properly sent.<br/>"
			                         "Thank you for your support, we will do our best to fix this "
			                         "problem as soon as possible.");
		}
	}

  private:
	std::string _minidump_path;
};
}

#endif // __PVGUIQT_PVCRASHREPORTER_H__
