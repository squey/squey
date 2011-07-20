#define SIMULATE_PIPELINE
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVElementFilterByFields.h>
#include <pvfilter/PVFieldsMappingFilter.h>
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
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file chunk_size" << endl;
		cerr << "Input must be a squid log file." << endl;
		return 1;
	}

	init_env();

	QCoreApplication app(argc, argv);

	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFilter::PVFieldsSplitter::p_type url_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("url");
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("regexp");
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>::p_type grep_lib_p = LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_one>)::get().get_filter_by_name("grep_regexp");

	if (!url_lib_p || !regexp_lib_p || !grep_lib_p) {
		cerr << "Unable to load one of the filters" << endl;
		return 1;
	}
	PVCore::PVArgumentList args;
	args["regexp"] = QRegExp("([0-9]+)[0-9.]*\\s+[0-9]+\\s+[0-9]+\\s+[A-Z/_-]+([0-9]+)\\s+[0-9]+\\s+(GET|POST|PUT|OPTIONS)\\s+(\\S+)\\s+(\\S+)\\s+([^/]+)/(\\d+.\\d+.\\d+.\\d+)");
	regexp_lib_p->set_args(args);
	args["regexp"] = QRegExp("(yahoo|lnc)");
	grep_lib_p->set_args(args);

	// Mapping filters
	
	// Mapping filter for the URL splitter
	PVFilter::PVFieldsMappingFilter::list_indexes indx;
	PVFilter::PVFieldsMappingFilter::map_filters mf;
	indx.push_back(3);
	mf[indx] = url_lib_p->f();
	PVFilter::PVFieldsMappingFilter mapping_url(mf);

	// Mapping filter for the URL splitter
	indx.clear();
	mf.clear();
	indx.push_back(4);
	mf[indx] = grep_lib_p->f();
	PVFilter::PVFieldsMappingFilter mapping_grep(mf);

	// Final composition
	PVFilter::PVFieldsBaseFilter_f f_final = boost::bind(mapping_grep.f(), boost::bind(mapping_url.f(), boost::bind(regexp_lib_p->f(), _1)));

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(f_final);
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	PVInput_p ifile(new PVInputFile(argv[1]));
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	return !process_filter(source, chk_flt->f());
}
