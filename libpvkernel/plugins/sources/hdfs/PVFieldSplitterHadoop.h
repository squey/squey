#ifndef PVFILTER_PVFIELDSPLITTERHADOOP_FILE_H
#define PVFILTER_PVFIELDSPLITTERHADOOP_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterHadoop : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterHadoop();

public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

private:
	CLASS_FILTER(PVFilter::PVFieldSplitterCSV)
};

}

#endif
