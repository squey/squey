#include <picviz/PVLayerFilter.h>
#include <picviz/PVLayerFilterAxisGradient.h>

#include <QRegExp>
#include <iostream>

using namespace std;

int main()
{
	REGISTER_FILTER(QString("Test"), Picviz::PVLayerFilterAxisGradient);

	LIB_FILTER(Picviz::PVLayerFilter)::list_filters const& l = LIB_FILTER(Picviz::PVLayerFilter)::get().get_list();
	LIB_FILTER(Picviz::PVLayerFilter)::list_filters::const_iterator it,ite;
	it = l.begin();
	ite = l.end();

	for (; it != ite; it++)
		cout << qPrintable(it.key()) << endl;

	return 0;
}
