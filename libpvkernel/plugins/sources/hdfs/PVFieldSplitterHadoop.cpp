#include "PVFieldSplitterCSV.h"


PVFilter::PVFieldSplitterHadoop::PVFieldSplitterHadoop()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterHadoop)

}
PVCore::list_fields::size_type PVFilter::PVFieldSplitterHadoop::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	return inf._nelts;
}


IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterHadoop)
