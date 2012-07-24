/**
 * \file pv_splitter.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PV_SPLITTER_H
#define PVCORE_PV_SPLITTER_H

#include <QString>
#include <QStringList>


#include <pvkernel/core/general.h>



class LibKernelDecl PVSplitter {
	private:
		QString name;

	public:
		PVSplitter(const QString &name_str);
		~PVSplitter();

		QString get_name()const{return name;}

};


#endif	/* PVCORE_PV_SPLITTER_H */
