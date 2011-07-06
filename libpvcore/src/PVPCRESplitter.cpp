/*
 * $Id: PVPCRESplitter.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */


#include <QString>
#include <QStringList>


#include <pvcore/debug.h>

#include <pvcore/pv_pcre_splitter.h>


PVPCRESplitter::PVPCRESplitter(const QString &name_str, const QString &regex_string) :
PVSplitter(name_str)
{
	q_regex = QRegExp(regex_string);
	regex = QString(regex_string);
}


PVPCRESplitter::~PVPCRESplitter()
{

}


QStringList PVPCRESplitter::apply(const QString &str)
{
	// VARIABLES
	QStringList output_list;

	// CODE
	if (q_regex.exactMatch(str)) {
		output_list = q_regex.capturedTexts();
		output_list.removeFirst();
	}

	return output_list;
}

