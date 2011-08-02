#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>

using namespace PVFilter;

std::list<PVFilter::PVFieldsFilterReg_p> container;
PVFieldsBaseFilter_f get_f()
{
	PVFilter::PVFieldsSplitter::p_type filter = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
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
