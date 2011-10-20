#ifndef PVRUSH_PVFILESERIALIZE_H
#define PVRUSH_PVFILESERIALIZE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <QString>

namespace PVCore {

// Helper class to serialize the original file if wanted
class LibKernelDecl PVFileSerialize
{
	friend class PVCore::PVSerializeObject;
public:
	PVFileSerialize(QString const& path);
public:
	QString const& get_path() const;

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _path;
};

}


#endif
