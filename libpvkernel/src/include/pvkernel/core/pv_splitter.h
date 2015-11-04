/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PV_SPLITTER_H
#define PVCORE_PV_SPLITTER_H

#include <QString>
#include <QStringList>


#include <pvkernel/core/general.h>



class PVSplitter {
	private:
		QString name;

	public:
		PVSplitter(const QString &name_str);
		~PVSplitter();

		QString get_name()const{return name;}

};


#endif	/* PVCORE_PV_SPLITTER_H */
