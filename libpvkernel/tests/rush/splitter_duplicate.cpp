/**
 * \file splitter_duplicate.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include <QCoreApplication>
#include "test-env.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVCore;

int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file chunk_size [separator]" << endl;
		return 1;
	}

	init_env();

	QCoreApplication app(argc, argv);

	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFilter::PVFieldsSplitter::p_type sp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("duplicate");
	if (!sp_lib_p) {
		cerr << "Unable to load splitter duplicate" << endl;
		return 1;
	}

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["n"] = 7;
	sp_lib_p->set_args(args);

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	PVInput_p ifile(new PVInputFile(argv[1]));
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	return !process_filter(source, chk_flt->f());
}