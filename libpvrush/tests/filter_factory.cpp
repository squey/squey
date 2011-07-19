#include <pvfilter/PVFieldSplitterRegexp.h>
#include <pvfilter/PVFieldFilterGrep.h>
#include <pvfilter/PVFieldsFilter.h>
#include <pvfilter/PVElementFilterGrep.h>
#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVFilterLibrary.h>
#include <pvrush/PVCSVSource.h>
#include <pvrush/PVUnicodeSource.h>

#include <QRegExp>
#include <iostream>

using namespace PVFilter;
using namespace std;

int main()
{
	REGISTER_FILTER(QString("PVFieldSplitterRegexp"), PVFilter::PVFieldSplitterRegexp);
	REGISTER_FILTER(QString("PVElementFilterGrep"), PVFilter::PVElementFilterGrep);

	{
		LIB_FILTER(PVElementFilter)::list_filters const& l = LIB_FILTER(PVElementFilter)::get().get_list();
		LIB_FILTER(PVElementFilter)::list_filters::const_iterator it,ite;
		it = l.begin();
		ite = l.end();

		for (; it != ite; it++)
			cout << qPrintable(it.key()) << endl;
	}

	{
		LIB_FILTER(PVFieldsBaseFilter::list_filters const& l1 = LIB_FILTER(PVFieldsBaseFilter)::get().get_list();
		LIB_FILTER(PVFieldsBaseFilter)::list_filters::const_iterator it1,ite1;
		it1 = l1.begin();
		ite1 = l1.end();

		for (; it1 != ite1; it1++)
			cout << qPrintable(it1.key()) << endl;
	}

	PVArgumentList args;
	args["regexp"] = QRegExp("^$");
	PVFieldSplitterRegexp::p_type re = PVFieldSplitterRegexp::p_type(new PVFieldSplitterRegexp(args));
	PVFieldSplitterRegexp::FilterT::p_type re2 = re->clone();

	cout << "First filter pointer: " << re.get() << endl;
	cout << "Second filter pointer: " << re2.get() << endl;

	return 0;
}
