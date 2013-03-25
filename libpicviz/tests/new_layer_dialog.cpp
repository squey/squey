/**
 * \file new_layer_dialog.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <iostream>

#include <QApplication>

#include <picviz/widgets/PVNewLayerDialog.h>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QString test = "toto";
	bool hide_layers = true;
	QString layer_name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(test, hide_layers);

	std::cout << "hide_layers=" << std::boolalpha << hide_layers << std::endl;
	std::cout << "layer_name=" << qPrintable(layer_name) << std::endl;

	return app.exec();
}
