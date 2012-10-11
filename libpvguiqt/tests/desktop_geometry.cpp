/**
 * \file desktop_geometry.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <iostream>

#include <pvkernel/core/lambda_connect.h>

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QPushButton>
#include <QDockWidget>

class CustomMainWindow : public QMainWindow
{
public:

	CustomMainWindow()
	{
		setGeometry(
		    QStyle::alignedRect(
		        Qt::LeftToRight,
		        Qt::AlignCenter,
		        size(),
		        qApp->desktop()->availableGeometry()
		    ));
	}
};

void print_infos()
{
	std::cout << "screen(-1)=" << QApplication::desktop()->screen(-1) << std::endl;
	std::cout << "screen(0)=" << QApplication::desktop()->screen(0) << std::endl;
	std::cout << "screen(1)=" << QApplication::desktop()->screen(1) << std::endl;
	std::cout << "isVirtualDesktop=" << QApplication::desktop()->isVirtualDesktop() << std::endl;
	std::cout << "screenCount=" << QApplication::desktop()->screenCount() << std::endl;
	std::cout << "screenNumber=" << QApplication::desktop()->screenNumber() << std::endl;

	QRect screenres = QApplication::desktop()->screenGeometry(0/*screenNumber*//*1*/);
	QDockWidget* secondDisplay = new QDockWidget(); // Use your QWidget
	secondDisplay->move(QPoint(screenres.x(), screenres.y()));
	secondDisplay->resize(screenres.width(), screenres.height());
	secondDisplay->show();
	//secondDisplay->showFullScreen();
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QPushButton* button = new QPushButton("Infos");
	CustomMainWindow* mw = new CustomMainWindow();
	mw->setCentralWidget(button);
	mw->show();

	connect(button, SIGNAL(clicked(bool)), []{print_infos();});

	return app.exec();
}
