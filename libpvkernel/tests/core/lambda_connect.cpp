/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/lambda_connect.h>

#include <QApplication>
#include <QMessageBox>
#include <QObject>
#include <QPushButton>


int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QPushButton* push_button = new QPushButton("lambda");
	connect(push_button, SIGNAL(clicked(bool)), [] {
		QMessageBox msgbox(QMessageBox::Question, "Lambda", "lambda_connect works! :)", QMessageBox::Ok);
		msgbox.exec();
	});

	push_button->show();

	return app.exec();
}
