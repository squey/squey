#include <pvfilter/PVPluginsLoad.h>
#include <pvfilter/PVFilterLibrary.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVFieldsFilter.h>

using namespace PVFilter;

std::list<PVFilter::PVFieldsFilterReg_p> container;
PVFieldsBaseFilter_f get_f()
{
	PVFilter::PVFieldsSplitter::p_type filter = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("regexp");
	PVFilter::PVFieldsFilterReg_p filter_clone = filter->clone<PVFilter::PVFieldsFilterReg>();
	container.push_back(filter_clone);
	return filter_clone->f();
}

int main()
{
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFieldsBaseFilter_f f = get_f();
	char* test = "salut";
	PVCore::PVField field(PVCore::PVElement(NULL, test, test+4), test, test+4);
	PVCore::list_fields lf;
	lf.push_back(field);
	f(lf);

	return 0;
}
