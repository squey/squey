#ifndef PVFILTER_PVSEQCHUNKFUNCTION_H
#define PVFILTER_PVSEQCHUNKFUNCTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVChunkFilter.h>

namespace PVFilter {

class LibKernelDecl PVSeqChunkFunction: public PVFilter::PVChunkFilter
{
	typedef PVCore::PVField::mapped_decimal_storage_type mapped_decimal_storage_type;
public:
	typedef std::function<void(PVCore::PVChunk*, const PVRow)> chunk_function_type;
	typedef std::vector<chunk_function_type> list_chunk_functions;

public:
	PVSeqChunkFunction():
		PVFilter::PVChunkFilter(),
		_cur_row(0)
	{ }

public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
	virtual func_type f() override { return boost::bind<PVCore::PVChunk*>((PVCore::PVChunk*(PVSeqChunkFunction::*)(PVCore::PVChunk*))(&PVSeqChunkFunction::operator()), this, _1); }

public:
	inline list_chunk_functions& chunk_functions() { return _chunk_funcs; }
	inline list_chunk_functions const& chunk_functions() const { return _chunk_funcs; }

private:
	list_chunk_functions _chunk_funcs;
	size_t _cur_row;

	//CLASS_FILTER_NONREG_NOPARAM(PVSeqChunkFunction)
	CLASS_FUNC_ARGS_NOPARAM(PVSeqChunkFunction)
};

}

#endif
