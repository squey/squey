/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVFILESERIALIZE_H
#define PVRUSH_PVFILESERIALIZE_H

#include <pvkernel/core/PVSerializeArchive.h>
#include <QString>

namespace PVCore
{

// Helper class to serialize the original file if wanted
class PVFileSerialize
{
	friend class PVCore::PVSerializeObject;

  public:
	PVFileSerialize(QString const& path);

  public:
	QString const& get_path() const;

  protected:
	void serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT

  protected:
	QString _path;
};
}

#endif
