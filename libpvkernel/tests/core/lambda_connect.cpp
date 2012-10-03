/**
 * \file lambda_connect.cpp
 *
 * Copyright (C) Picviz Labs 2012
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
