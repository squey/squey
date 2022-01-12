/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
		send_report->setEnabled(false); // disable send button until API is up again
		QPushButton* cancel = new QPushButton("Don't send");

		QDialogButtonBox* button_box = new QDialogButtonBox();
		button_box->addButton(send_report, QDialogButtonBox::AcceptRole);
		button_box->addButton(cancel, QDialogButtonBox::RejectRole);

		connect(button_box, &QDialogButtonBox::accepted, this, &PVCrashReporterDialog::send_crash);
		connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

		QVBoxLayout* main_layout = new QVBoxLayout(this);

		QLabel* text = new QLabel(QString(
		    "<b>The program has crashed and we apologize for the inconvenience.</b><br/><br/>"
		    "Please send us the crash report so we can diagnose and fix the problem.<br/>"
		    "We will treat this report as confidential and only to improve this "
		    "product.<br/><br/>"
			"The generated crash report is locally stored at the following path : <br/>%1").arg(minidump_path.c_str()));
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
		std::string locking_code = "";
		int ret = PVCore::PVCrashReportSender::send(_minidump_path, INENDI_CURRENT_VERSION_STR,
		                                            locking_code);
		if (ret == 413) { // Payload Too Large
			QMessageBox::critical(this, "Error sending crash report",
			                      "The crash report size exceeded the server accepted "
			                      "size.<br>Please, send it by file sharing.");
		} else if (ret != 0) {
			QMessageBox::critical(
			    this, "Error sending crash report",
			    QString("Please, check your Internet connection (HTTP status %1) ").arg(ret));
		} else {
			QMessageBox::information(this, "Crash report sent with success",
			                         "Your crash report was properly sent.<br/>"
			                         "Thank you for your support, we will do our best to fix this "
			                         "problem as soon as possible.");
		}

		accept();
	}

  private:
	std::string _minidump_path;
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVCRASHREPORTER_H__
