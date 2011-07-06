#define SIMULATE_PIPELINE
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVElementFilterByFields.h>
#include <pvfilter/PVPluginsLoad.h>
#include <pvrush/PVInputFile.h>
#include <pvrush/PVUnicodeSource.h>
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
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " file chunk_size regexp" << endl;
		return 1;
	}

	init_env();

	QCoreApplication app(argc, argv);

	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFilter::PVFieldsSplitter::p_type sp_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("regexp");
	if (!sp_lib_p) {
		cerr << "Unable to load splitter regexp" << endl;
		return 1;
	}

	PVFilter::PVArgumentList args;
	args["regexp"] = PVFilter::PVArgument(QRegExp(QString(argv[3])));
	sp_lib_p->set_args(args);

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	PVInputFile ifile(argv[1]);
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	return !process_filter(source, chk_flt->f());
}
