#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVChunk.h>
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

#include <stdio.h>
#include <stdlib.h>
#include "test-env.h"

#include <valgrind/callgrind.h>

#define NLINES 10000000

#define NTRIES 6

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

// Virtual source that only creates elements
template < template <class T> class Allocator = PVCore::PVMMapAllocator >
class PVElementsSource: public PVRush::PVRawSourceBase
{
	typedef PVCore::PVChunkMem<Allocator> PVChunkAlloc;
	typedef Allocator<char> alloc_chunk;
public:
	PVElementsSource(size_t nchunks, size_t size_chunk, size_t nelts_chunk):
		PVRawSourceBase(_null_filter.f()),
		_nchunks(nchunks),
		_size_chunk(size_chunk),
		_nelts_chunk(nelts_chunk)
	{
		_cur_chunk = 0;
	}

public:
	virtual QString human_name() { return "memory footprint source"; }
	virtual void seek_begin() { }
	virtual bool seek(PVRush::input_offset /*off*/) { return true; }
	virtual void prepare_for_nelts(chunk_index /*nelts*/) { }
	virtual PVCore::PVChunk* operator()()
	{
		if (_cur_chunk >= _nchunks) {
			_cur_chunk = 0;
			return NULL;
		}
		PVCore::PVChunk* chunk = PVChunkAlloc::allocate(_size_chunk, this, _alloc);
		memset(chunk->begin(), 0, chunk->avail());
		for (size_t i = 0; i < _nelts_chunk; i++) {
			chunk->add_element(chunk->begin(), chunk->physical_end());
		}
		chunk->init_elements_fields();
		PVLOG_INFO("Chunk %d created\n", _cur_chunk);
		_cur_chunk++;
		return chunk;
	}

	virtual PVRush::input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx)
	{
		known_idx = idx;
		return idx;
	}

	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVElementsSource<Allocator>::operator(), this); }
private:
	PVFilter::PVChunkFilter _null_filter;
	alloc_chunk _alloc;
	size_t _nchunks;
	size_t _size_chunk;
	size_t _nelts_chunk;
	size_t _cur_chunk;
};

void bench(PVRush::PVExtractor &ext, size_t nlines)
{
	ext.start_controller();
	//tbb::tick_count start = tbb::tick_count::now();
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, nlines);
	job->wait_end();
	//tbb::tick_count end = tbb::tick_count::now();
	ext.force_stop_controller();
}

void bench(size_t nchunks, size_t size_chunk, size_t neltsperc, size_t nfields)
{
	// Create fake source
	PVFieldCreator filter_fc(nfields);
	PVFilter::PVElementFilterByFields felt(filter_fc.f());
	PVFilter::PVChunkFilterByElt fchunk(felt.f());
	PVRush::PVRawSourceBase_p src(new PVElementsSource<>(nchunks, size_chunk, neltsperc));

	PVRush::PVExtractor ext(1);
	ext.add_source(src);

	ext.set_chunk_filter(fchunk);
	ext.force_number_axes(nfields);

	bench(ext, nchunks*neltsperc);

	PVLOG_DEBUG("Extraction finshed. Press a key to delete the NRAW.\n");
	getchar();
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " size_chunk nchunks neltsperchunk nfields " << std::endl;
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
	size_t size_chunk = atoll(argv[1]);
	size_t nchunks = atoll(argv[2]);
	size_t neltsperc = atoll(argv[3]);
	size_t nfields = atoll(argv[4]);

	// Memory footprint of the TBB pipeline.
	// This is done by creating lots of fields and elements, and see how much memory is "wasted" by the whole process (allocator, etc...)
	std::cout << "Press a key to start memory benchmark..." << std::endl;
	getchar();
	bench(nchunks, size_chunk, neltsperc, nfields);
	std::cout << "End of memory benchmark. Press a key to exit this program." << std::endl;
	getchar();

	return 0;
}
