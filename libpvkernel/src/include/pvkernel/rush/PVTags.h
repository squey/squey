/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVTAG_H
#define PVCORE_PVTAG_H

#include <QSet>
#include <QString>

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

namespace PVRush
{
class PVTags
{

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

#endif /* PVCORE_PVTAG_H */
