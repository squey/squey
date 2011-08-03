#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVPluginsLoad.h>

#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

#include <QString>
#include <QStringList>

#include <iostream>

#include <stdlib.h>

#define NLINES 10000000
#define NCHUNKS 100

#define NTRIES 4

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

double bench(PVRush::PVExtractor &ext, size_t nlines)
{
	fflush(stdout);
	ext.start_controller();
	double dur = 0;
	for (int i = 0; i < NTRIES; i++) {
		clear_disk_cache();
		tbb::tick_count start = tbb::tick_count::now();
		PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, nlines);
		job->wait_end();
		tbb::tick_count end = tbb::tick_count::now();
		if (job->done()) {
			printf("Warning: all lines have been read (which mean that, maybe, not all the datas have been processed !\n");
		}
		dur += (end-start).seconds();
	}
	ext.force_stop_controller();
	return dur/NTRIES;
}

double bench_utf16_align(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines)
{
	PVRush::PVExtractor ext(NCHUNKS);
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVRawSourceBase_p src(new PVRush::PVUnicodeSource<>(ifile, chunk_size, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines);
}

double bench_utf16(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines)
{
	PVRush::PVExtractor ext(NCHUNKS);
	PVRush::PVChunkAlign null_align;
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVChunkTransformUTF16* utf16 = new PVRush::PVChunkTransformUTF16();
		PVRush::PVRawSourceBase_p src(new PVRush::PVRawSource<>(ifile, null_align, chunk_size, *utf16, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines);
}

double bench(QStringList const& paths, PVFilter::PVChunkFilter_f chk_flt, size_t chunk_size, size_t nlines)
{
	PVRush::PVExtractor ext(NCHUNKS);
	PVRush::PVChunkAlign null_align;
	PVRush::PVChunkTransform null_transform;
	for (int i = 0; i < paths.size(); i++) {
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(paths[i].toLocal8Bit().constData()));
		PVFilter::PVChunkFilter* chk_flt_src = new PVFilter::PVChunkFilter();
		PVRush::PVRawSourceBase_p src(new PVRush::PVRawSource<>(ifile, null_align, chunk_size, null_transform, chk_flt_src->f()));
		ext.add_source(src);
	}
	ext.set_chunk_filter(chk_flt);
	return bench(ext, nlines);
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


inline void print_perf(double dur, size_t size)
{
	printf(" throuhput: %0.4f MB/s.\n", (double)(size)/(1024*1024*dur));
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " chunk_size file [file] [file ...]" << std::endl;
		return 1;
	}

	int nfiles = argc-2;
	char** files = &argv[2];
	QStringList lfiles;
	for (int i = 0; i < nfiles; i++) {
		lfiles << QString(files[i]);
	}

	int chunk_size = atoi(argv[1]);
	if (chunk_size < 1024) {
		chunk_size = 1024;
	}

	PVFilter::PVChunkFilter chk_flt_null;
	tbb::tick_count::interval_t durd;
	
	tbb::task_scheduler_init init(1);

	PVFilter::PVPluginsLoad::load_all_plugins();

	clear_disk_cache();


	// Disk reading performance
	printf("Disk reading performance..");
	tbb::tick_count start = tbb::tick_count::now();
	size_t total_read = read_files(files, nfiles, chunk_size);
	tbb::tick_count end = tbb::tick_count::now();
	durd = end-start;
	printf(" read %0.4f MB in %0.4f s", (double)(total_read)/(1024*1024), durd.seconds());
	print_perf(durd.seconds(), total_read);

	double dur;	
	// Serial reading performance with no transformation ("architecture" overhead)
//	printf("Serial reading performance with no transformation (\"architecture\" overhead)");
//	dur = bench(lfiles, chk_flt_null.f(), chunk_size, NLINES);
//	print_perf(dur, total_read);
//	clear_disk_cache();

//	// Serial reading with UTF16 transformation
	printf("Serial reading with UTF16 transformation");
	dur = bench_utf16(lfiles, chk_flt_null.f(), chunk_size, NLINES);
	print_perf(dur, total_read);
	
	// Serial reading with UTF16 transformation and alignement
	printf("Serial reading with UTF16 transformation and alignement");
	dur = bench_utf16_align(lfiles, chk_flt_null.f(), chunk_size, NLINES);
	print_perf(dur, total_read);
	
	// Parallel URL splitter only
	
	// Parallel regexp splitter only
	printf("Parallel regexp splitter only");
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
	PVFilter::PVFieldsBaseFilter_p fre_in = regexp_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QRegExp("^a$"));
	fre_in->set_args(args);
	PVFilter::PVElementFilterByFields felt(fre_in->f());
	PVFilter::PVChunkFilterByElt fchunk(felt.f());

	dur = bench_utf16_align(lfiles, fchunk.f(), chunk_size, NLINES);
	print_perf(dur, total_read);


	
	// Parallel regexp grep only
	

	return 0;
}
