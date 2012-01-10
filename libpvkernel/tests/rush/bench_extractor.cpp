#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVPluginsLoad.h>

#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

#include <QCoreApplication>
#include <QString>
#include <QStringList>

#include <iostream>
#include <math.h>

#include <stdlib.h>
#include "test-env.h"

#include <valgrind/callgrind.h>

#define NLINES 10000000

#define NTRIES 1

class PVFieldCreator: public PVFilter::PVFieldsSplitter
{
public:
	PVFieldCreator(int nparams):
		PVFilter::PVFieldsSplitter(),
		_nparams(nparams)
	{
		INIT_FILTER_NOPARAM(PVFieldCreator);
		_nparams = nparams;
	}
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
	{
		PVCore::PVField nf(field);
		for (int i = 0; i < _nparams; i++) {
			l.insert(it_ins, nf);
		}
		return _nparams;
	}
public:
	void set_nfields(int nparams) { _nparams = nparams; }
protected:
	int _nparams;
	CLASS_FILTER_NOPARAM(PVFieldCreator)
};

class PVFieldProcess: public PVFilter::PVFieldsSplitter
{
public:
	PVFieldProcess():
		PVFilter::PVFieldsSplitter()
	{
		INIT_FILTER_NOPARAM(PVFieldCreator);
	}
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
	{
		UChar* begin = (UChar*) field.begin();
		size_t n = field.size()/sizeof(UChar);
		for (size_t i = 0; i < n; i++) {
			begin[i] = ((begin[i] + 1) << 4)*8;
		}
		return 0;
	}
protected:
	
	CLASS_FILTER_NOPARAM(PVFieldProcess)
};

void clear_disk_cache()
{
	//std::cerr << "Clearing disk caches...";
	// Clear disk cache in order to have real results !
#ifndef WIN32
	system("sync");
	system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'");
#endif
	//std::cerr << " done." << std::endl;
}

double bench(PVRush::PVExtractor &ext, size_t nlines, size_t nthreads)
{
	fflush(stdout);
	ext.force_number_axes(7);
	ext.start_controller();
	double dur = 0;
	std::vector<double> durs;
	durs.reserve(NTRIES);
	for (int i = 0; i < NTRIES; i++) {
		//clear_disk_cache();
		tbb::tick_count start = tbb::tick_count::now();
		PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, nlines);
		job->wait_end();
		tbb::tick_count end = tbb::tick_count::now();
		if (job->done()) {
			printf("Warning: all lines have been read (which mean that, maybe, not all the datas have been processed !\n");
		}
		durs.push_back((end-start).seconds());
	}
	ext.force_stop_controller();
	std::sort(durs.begin(), durs.end());
	printf(" NRAW: %u lines, %u fields.\n", ext.get_nraw().get_number_rows(), ext.get_nraw().get_number_cols());
	return durs[NTRIES/2];
}

double bench_utf16_align(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines, size_t nchunks, size_t nthreads = 0)
{
	PVRush::PVExtractor ext(nchunks);
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVRawSourceBase_p src(new PVRush::PVUnicodeSource<>(ifile, chunk_size, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines, nthreads);
}

double bench_utf16(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines, size_t nchunks, size_t nthreads = 0)
{
	PVRush::PVExtractor ext(nchunks);
	PVRush::PVChunkAlign null_align;
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVChunkTransformUTF16* utf16 = new PVRush::PVChunkTransformUTF16();
		PVRush::PVRawSourceBase_p src(new PVRush::PVRawSource<>(ifile, null_align, chunk_size, *utf16, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines, nthreads);
}

double bench(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines, size_t nchunks, size_t nthreads = 0)
{
	PVRush::PVExtractor ext(nchunks);
	PVRush::PVChunkAlign null_align;
	PVRush::PVChunkTransform null_transform;
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVRawSourceBase_p src(new PVRush::PVRawSource<>(ifile, null_align, chunk_size, null_transform, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines, nthreads);
}

size_t read_files(char** files, int n, int chunk_size)
{
	char* buf = (char*) malloc(chunk_size);
	size_t ret = 0;

	// fopen/fread is used because this is what PVInputFile uses indirectly.
	for (int i = 0; i < n; i++) {
		FILE* f = fopen(files[i], "r");
		size_t r;
		while ((r = fread(buf, 1, chunk_size, f)) > 0) {
			ret += r;
		}
		fclose(f);
	}

	free(buf);

	return ret;
}


#define CALC_BW(size, dur) (((double)((size)))/(1024*1024*(dur)))
inline void print_perf(double dur, size_t size)
{
	printf(" throuhput: %0.4f MB/s.\n", CALC_BW(size, dur));
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " nchunks chunk_size only_rx file [file] [file ...]" << std::endl;
		return 1;
	}
	QCoreApplication(argc, argv);
	init_env();
#ifdef __SSE4_1__
	std::cout << "Tests compiled w/ SSE4.1." << std::endl;
	if (PVCore::PVIntrinsics::has_sse41()) {
		std::cout << "SSE4.1 enabled." << std::endl;
	}
#endif

	int nfiles = argc-4;
	char** files = &argv[4];
	QStringList lfiles;
	for (int i = 0; i < nfiles; i++) {
		lfiles << QString(files[i]);
	}

	int chunk_size = atoi(argv[2]);
	if (chunk_size < 1024) {
		chunk_size = 1024;
	}

	int nchunks = atoi(argv[1]);
	if (nchunks < 24) {
		nchunks = 24;
	}

	bool only_rx = atoi(argv[3]) == 1;

	PVFilter::PVChunkFilter chk_flt_null;
	tbb::tick_count::interval_t durd;

	PVFilter::PVPluginsLoad::load_all_plugins();

	// Disk reading performance
	printf("Disk reading performance..");
	clear_disk_cache();
	tbb::tick_count start = tbb::tick_count::now();
	size_t total_read = read_files(files, nfiles, chunk_size);
	tbb::tick_count end = tbb::tick_count::now();
	durd = end-start;
	printf(" read %0.4f MB in %0.4f s", (double)(total_read)/(1024*1024), durd.seconds());
	print_perf(durd.seconds(), total_read);

	//clear_disk_cache();

	double dur;	

	// Regexp setup
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
	PVFilter::PVFieldsBaseFilter_p fre_in = regexp_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QString("(.*) (.*)"));
	fre_in->set_args(args);
	PVFilter::PVElementFilterByFields felt(fre_in->f());
	PVFilter::PVChunkFilterByElt fchunk(felt.f());

	if (!only_rx) {
		// Serial reading performance with no transformation ("architecture" overhead)
		printf("Serial reading performance with no transformation (\"architecture\" overhead)");
		dur = bench(lfiles, chk_flt_null.f(), chunk_size, NLINES, nchunks);
		print_perf(dur, total_read);
		// Serial reading with UTF16 transformation
		printf("Serial reading with UTF16 transformation");
		CALLGRIND_START_INSTRUMENTATION
			dur = bench_utf16(lfiles, chk_flt_null.f(), chunk_size, NLINES, nchunks);
		CALLGRIND_STOP_INSTRUMENTATION
			print_perf(dur, total_read);

		// Serial reading with UTF16 transformation and alignement
		printf("Serial reading with UTF16 transformation and alignement");
		dur = bench_utf16_align(lfiles, chk_flt_null.f(), chunk_size, NLINES, nchunks);
		print_perf(dur, total_read);

#if 0
		// Field creation
		printf("Parallel field creation (2)");
		PVFieldCreator filter_fc(2);
		PVFilter::PVElementFilterByFields felt2(filter_fc.f());
		PVFilter::PVChunkFilterByElt fchunk2(felt2.f());
		dur = bench_utf16_align(lfiles, fchunk2.f(), chunk_size, NLINES, nchunks);
		print_perf(dur, total_read);

		for (int i = 2; i < 10; i++) {
			filter_fc.set_nfields(i*2);
			printf("Parallel field creation (%d)", i*2);
			dur = bench_utf16_align(lfiles, fchunk2.f(), chunk_size, NLINES, nchunks);
			print_perf(dur, total_read);
		}
#endif

		// Parallel URL splitter only

		// Parallel regexp splitter only
		printf("Parallel simple regexp splitter only");

		dur = bench_utf16_align(lfiles, fchunk.f(), chunk_size, NLINES, nchunks);
		print_perf(dur, total_read);
	}

#if 0
	size_t chunk_sizesK[] = {600,800,1024,1200,1400,1600};
	PVFieldProcess filter_pr;
	PVFilter::PVElementFilterByFields felt2(filter_pr.f());
	PVFilter::PVChunkFilterByElt fchunk2(felt2.f());
	for (size_t i = 0; i < sizeof(chunk_sizesK)/sizeof(size_t); i++) {
		//size_t nchunks_arr[] = {48,72,92,108,132,156,180,240};
		size_t nchunks_arr[] = {6,12,24,48,72,92};
		for (size_t j = 0; j < sizeof(nchunks_arr)/sizeof(size_t); j++) {
			dur = bench(lfiles, fchunk2.f(), chunk_sizesK[i], NLINES, nchunks_arr[j], nthreads);
			printf("%lu %lu %0.6f\n", chunk_sizesK[i], nchunks_arr[j], CALC_BW(total_read, dur));
		}
	}
#endif

	printf("Parallel squid regexp splitter only");
	//args["regexp"] = PVCore::PVArgument(QString("(\\d+.\\d+.\\d+.\\d+) \\- \\S+ \\[(\\d+\\/\\w+\\/\\d+:\\d+:\\d+:\\d+).*\\] \\\"(\\w+) (.*) (\\S+)\\\" (\\d+) (.*)"));
	args["regexp"] = PVCore::PVArgument(QString("([0-9]+)[0-9.]*\\s+[0-9]+\\s+[0-9]+\\s+[A-Z/_-]+([0-9]+)\\s+[0-9]+\\s+(GET|POST|PUT|OPTIONS)\\s+(\\S+)\\s+(\\S+)\\s+([^/]+)/(\\d+.\\d+.\\d+.\\d+)"));
	fre_in->set_args(args);
	dur = bench_utf16_align(lfiles, fchunk.f(), chunk_size, NLINES, nchunks);
	print_perf(dur, total_read);
	// Parallel regexp grep only
	

	return 0;
}
