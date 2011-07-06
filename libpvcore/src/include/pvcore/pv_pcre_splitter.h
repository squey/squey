/*
 * $Id: pv_pcre_splitter.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#ifndef PVCORE_PV_PCRE_SPLITTER_H
#define PVCORE_PV_PCRE_SPLITTER_H

#include <QRegExp>
#include <QString>
#include <QStringList>


#include <pvcore/general.h>

#include <pvcore/pv_splitter.h>


class LibExport PVPCRESplitter : public PVSplitter {
	private:
		QRegExp q_regex;
		QString regex;
		

	public:
		PVPCRESplitter(const QString &name_str, const QString &regex_string);
		~PVPCRESplitter();

		QStringList apply(const QString &str);

};


#endif	/* PVCORE_PV_PCRE_SPLITTER_H */
