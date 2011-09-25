/*
 * $Id: pv_axis_format.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_PVTAG_H
#define PVCORE_PVTAG_H

#include <QSet>
#include <QString>

#include <pvkernel/core/general.h>

/**
 * \class PVRush::Tag
 * \defgroup Define a list of tags
 * \brief Tags are used to strictly define the bahavior of a given axis regardless its real name.
 * @{
 *
* Tags are put in a PVAxisFormat to define a behavior in the different part of the software that
* need to know what can or cannot do a given axis.
 *
 */

namespace PVRush {
class LibKernelDecl PVTags {

 private:
	QSet<QString> _tags;

 public:
	PVTags();
	~PVTags();
	
	void add_tag(QString tag);
	bool del_tag(QString tag);
	bool has_tag(QString tag) const;
	QSet<QString> const& list() const { return _tags; }
};
}

/*@}*/

#endif	/* PVCORE_PVTAG_H */
