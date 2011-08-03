#include <pvkernel/filter/PVFieldSplitterRegexp.h>
#include <pvkernel/filter/PVFieldFilterGrep.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterGrep.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/rush/PVCSVSource.h>
#include <pvkernel/rush/PVUnicodeSource.h>

#include <QRegExp>
#include <iostream>

using namespace PVFilter;
using namespace std;

int main()
{
	REGISTER_CLASS(QString("PVFieldSplitterRegexp"), PVFilter::PVFieldSplitterRegexp);
	REGISTER_CLASS(QString("PVElementFilterGrep"), PVFilter::PVElementFilterGrep);

	{
		LIB_CLASS(PVElementFilter)::list_filters const& l = LIB_CLASS(PVElementFilter)::get().get_list();
		LIB_CLASS(PVElementFilter)::list_filters::const_iterator it,ite;
		it = l.begin();
		ite = l.end();

		for (; it != ite; it++)
			cout << qPrintable(it.key()) << endl;
	}

	{
		LIB_CLASS(PVFieldsBaseFilter::list_filters const& l1 = LIB_CLASS(PVFieldsBaseFilter)::get().get_list();
		LIB_CLASS(PVFieldsBaseFilter)::list_filters::const_iterator it1,ite1;
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
