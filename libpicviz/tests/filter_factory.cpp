/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <picviz/PVLayerFilter.h>
#include <picviz/PVLayerFilterAxisGradient.h>

#include <QRegExp>
#include <iostream>

using namespace std;

int main()
{
	REGISTER_FILTER(QString("Test"), Picviz::PVLayerFilterAxisGradient);

	LIB_CLASS(Picviz::PVLayerFilter)::list_filters const& l = LIB_CLASS(Picviz::PVLayerFilter)::get().get_list();
	LIB_CLASS(Picviz::PVLayerFilter)::list_filters::const_iterator it,ite;
	it = l.begin();
	ite = l.end();

	for (; it != ite; it++)
		cout << qPrintable(it.key()) << endl;

	return 0;
}
