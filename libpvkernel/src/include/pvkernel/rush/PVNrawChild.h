/*
 * $Id: PVNrawChild.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#ifndef PVRUSH_NRAWCHILD_H
#define PVRUSH_NRAWCHILD_H

#include <QString>
#include <QStringList>
#include <QVector>

#include <pvkernel/core/general.h>

#include <pvkernel/rush/PVFormat.h>

namespace PVRush {

	class PVNraw;

	class LibKernelDecl PVNrawChild {
	private:
		PVNraw *parent;
		PVRow *parent_line_map;
	public:
		//PVNrawChild(PVNraw *parent_);
		//~PVNrawChild();

		QVector<QStringList> table;		
		PVFormat format;

		void append(QStringList list);
		QString get_value(PVRow row, PVCol col);
	};

}

#endif	/* PVRUSH_NRAWCHILD_H */
