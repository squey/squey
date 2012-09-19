/**
 * \file css.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <iostream>

#include <QApplication>
#include <QMainWindow>
#include <QFile>
#include <QTextStream>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QLabel>
#include <QStyle>
#include <QDesktopWidget>

class CustomMainWindow : public QMainWindow
{
public:

	CustomMainWindow()
	{
		setMinimumSize(500,600);

		setGeometry(
		    QStyle::alignedRect(
		        Qt::LeftToRight,
		        Qt::AlignCenter,
		        size(),
		        qApp->desktop()->availableGeometry()
		    ));

		load_stylesheet();
	}

	void keyPressEvent(QKeyEvent *event)
	{
		switch (event->key()) {
			case Qt::Key_Dollar:
			{
				load_stylesheet();
				break;
			}
		}
	}

	void load_stylesheet()
	{
		std::cout << "Refresh CSS" << std::endl;
		QFile css_file("css.css");
		css_file.open(QFile::ReadOnly);
		QTextStream css_stream(&css_file);
		QString css_string(css_stream.readAll());
		css_file.close();

		setStyleSheet(css_string);
		setStyle(QApplication::style());QVBoxLayout* layout2 = new QVBoxLayout();
	}
};

int main(int argc, char** argv)
{
	// Qt app
	QApplication app(argc, argv);

	QMainWindow* mw = new CustomMainWindow();

	QWidget* main_widget = new QWidget();
	main_widget->setObjectName("CSSMainWidget");

	QWidget* widget1 = new QWidget(main_widget);
	widget1->setObjectName("CSSWidget1");

	// Note: Setting layout to a widget change the parent of the widget
	QVBoxLayout* main_layout = new QVBoxLayout(widget1);
	std::cout << "main_layout parent is widget1:" << std::boolalpha << (main_layout->parent() ==  widget1) << std::endl;
	std::cout << "(setting main_layout to main_widget)" << std::endl;
	main_widget->setLayout(main_layout);
	std::cout << "main_layout parent is main_widget:" << std::boolalpha << (main_layout->parent() ==  main_widget) << std::endl;

	QLabel* label1 = new QLabel("Test1", widget1);

	// Note: Adding a widget to a layout change the parent of the widget
	std::cout << "label1 parent is widget1:" << std::boolalpha << (label1->parent() ==  widget1) << std::endl;
	std::cout << "(adding label1 to main_layout)" << std::endl;
	main_layout->addWidget(label1);
	std::cout << "label1 parent is widget1:" << std::boolalpha << (label1->parent() ==  widget1) << std::endl;
	std::cout << "label1 parent is main_widget:" << std::boolalpha << (label1->parent() ==  main_widget) << std::endl;

	// Widget2
	QWidget* widget2 = new QWidget();
	widget2->setObjectName("CSSWidget2");
	QVBoxLayout* layout1 = new QVBoxLayout();
	widget2->setLayout(layout1);
	layout1->addWidget(new QLabel("Test2"));

	// Widget3
	QWidget* widget3 = new QWidget();
	widget3->setObjectName("CSSWidget3");
	QVBoxLayout* layout2 = new QVBoxLayout();
	widget3->setLayout(layout2);

	// Note: several widgets can have the same object name, thus allowing to simulate a CSS class behavior.
	QLabel* label3 = new QLabel("Test3");
	label3->setObjectName("CSSClassLabel");
	QLabel* label4 = new QLabel("Test4");
	label4->setObjectName("CSSClassLabel");
	layout2->addWidget(label3);
	layout2->addWidget(label4);

	// main_layout parent is main_widget, so widget2 and widget3 respectively match "CSSWidget2" and "CSSWidget3" but also "CSSMainWidget".
	main_layout->addWidget(widget2);
	main_layout->addWidget(widget3);

	mw->setCentralWidget(main_widget);
	mw->show();

	return app.exec();
}
