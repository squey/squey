/**
 * \file pv_pcre_splitter.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PV_PCRE_SPLITTER_H
#define PVCORE_PV_PCRE_SPLITTER_H

#include <QRegExp>
#include <QString>
#include <QStringList>


#include <pvkernel/core/general.h>

#include <pvkernel/core/pv_splitter.h>


class LibKernelDecl PVPCRESplitter : public PVSplitter {
	private:
		QRegExp q_regex;
		QString regex;
		

	public:
		PVPCRESplitter(const QString &name_str, const QString &regex_string);
		~PVPCRESplitter();

		QStringList apply(const QString &str);

};


#endif	/* PVCORE_PV_PCRE_SPLITTER_H */
