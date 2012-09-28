#ifndef PVFILTER_PVPUREMAPPINGPROCESSING_H
#define PVFILTER_PVPUREMAPPINGPROCESSING_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVChunkFilter.h>

namespace PVFilter {

class LibKernelDecl PVPureMappingProcessing: public PVFilter::PVChunkFilter
{
	typedef PVCore::PVField::mapped_decimal_storage_type mapped_decimal_storage_type;
public:
	typedef std::function<mapped_decimal_storage_type(PVCore::PVField const& field)> pure_mapping_func;
	typedef std::vector<pure_mapping_func> list_pure_mapping_t;

public:
	PVPureMappingProcessing():
		PVFilter::PVChunkFilter()
	{ }

public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
	virtual func_type f() override { return boost::bind<PVCore::PVChunk*>((PVCore::PVChunk*(PVPureMappingProcessing::*)(PVCore::PVChunk*))(&PVPureMappingProcessing::operator()), this, _1); }

public:
	list_pure_mapping_t& pure_mappings() { return _mappings; }
	list_pure_mapping_t const& pure_mappings() const { return _mappings; }

private:
	list_pure_mapping_t _mappings;

	//CLASS_FILTER_NONREG_NOPARAM(PVPureMappingProcessing)
	CLASS_FUNC_ARGS_NOPARAM(PVPureMappingProcessing)
};

}

#endif
