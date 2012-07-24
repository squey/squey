/**
 * \file splitter_pcap.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#define SIMULATE_PIPELINE
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputPcap.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/rush/PVRawSource.h>
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
		cerr << "Usage: " << argv[0] << " file chunk_size" << endl;
		return 1;
	}

	init_env();

	QCoreApplication app(argc, argv);

	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFilter::PVFieldsSplitter::p_type sp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("pcap");
	if (!sp_lib_p) {
		cerr << "Unable to load splitter PCAP" << endl;
		return 1;
	}

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	PVInput_p ifile(new PVInputPcap(argv[1]));
	PVFilter::PVChunkFilter null;
	PVRush::PVChunkAlign align;
	PVRush::PVChunkTransform transform;
	PVRush::PVRawSource<> source(ifile, align, atoi(argv[2]), transform, null);

	return !process_filter(source, chk_flt->f());
}
