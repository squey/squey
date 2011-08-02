/*
 * $Id: pv_splitter.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
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
