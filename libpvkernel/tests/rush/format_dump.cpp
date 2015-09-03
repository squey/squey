/**
 * \file format_dump.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVInputFile.h>

#include <pvkernel/filter/PVPluginsLoad.h>

#include <QDir>
#include <QStringList>

#include <iostream>

#include "helpers.h"
#include "test-env.h"

void dump_args(PVRush::PVAxisFormat::node_args_t const& args)
{
	PVRush::PVAxisFormat::node_args_t::const_iterator it;
	for (it = args.begin(); it != args.end(); it++) {
		std::cout << qPrintable(it.key()) << "=" << qPrintable(it.value()) << std::endl;
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " format" << std::endl;
		return 1;
	}

	// Initialisation
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	// Format reading

	QString fpath = argv[1];
	PVRush::PVFormat format("format", fpath);
	std::cerr << "Start populating the format..." << std::endl;
	if (!format.populate(true)) {
		std::cerr << "Unable to populate format from file " << qPrintable(fpath) << std::endl;
		return 1;
	}

	format.debug();

	PVRush::list_axes_t const& axes = format.get_axes();
	PVRush::list_axes_t::const_iterator it;
	PVCol axis_id = 0;
	for (it = axes.begin(); it != axes.end(); it++) {
		PVRush::PVAxisFormat::node_args_t const& args = it->get_args_mapping_string();
		std::cout << "For axis " << axis_id << ", mapping args are :" << std::endl;
		dump_args(args);
		std::cout << "For axis " << axis_id << ", plotting args are :" << std::endl;
		PVRush::PVAxisFormat::node_args_t const& args_p = it->get_args_plotting_string();
		dump_args(args_p);
		axis_id++;
	}


	return 0;
}
