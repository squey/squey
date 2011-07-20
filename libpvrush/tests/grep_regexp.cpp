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
		cerr << "Usage: " << argv[0] << " file chunk_size regexp reverse" << endl;
		return 1;
	}

	bool reverse = (argc < 5) ? false : (argv[4][0] == '1');

	init_env();

	QCoreApplication app(argc, argv);

	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>::p_type sp_lib_p = LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_one>)::get().get_filter_by_name("grep_regexp");
	if (!sp_lib_p) {
		cerr << "Unable to load filter regexp" << endl;
		return 1;
	}

	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QRegExp(QString(argv[3])));
	args["reverse"] = reverse;
	sp_lib_p->set_args(args);

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	 PVInput_p ifile(new PVInputFile(argv[1]));
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	return !process_filter(source, chk_flt->f());
}
