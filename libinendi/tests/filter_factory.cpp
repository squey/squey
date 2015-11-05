/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayerFilter.h>
#include <inendi/PVLayerFilterAxisGradient.h>

#include <QRegExp>
#include <iostream>

using namespace std;

int main()
{
	REGISTER_FILTER(QString("Test"), Inendi::PVLayerFilterAxisGradient);

	LIB_CLASS(Inendi::PVLayerFilter)::list_filters const& l = LIB_CLASS(Inendi::PVLayerFilter)::get().get_list();
	LIB_CLASS(Inendi::PVLayerFilter)::list_filters::const_iterator it,ite;
	it = l.begin();
	ite = l.end();

	for (; it != ite; it++)
		cout << qPrintable(it.key()) << endl;

	return 0;
}
