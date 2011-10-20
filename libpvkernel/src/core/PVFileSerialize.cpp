#include <pvkernel/core/PVFileSerialize.h>


PVCore::PVFileSerialize::PVFileSerialize(QString const& path):
	_path(path)
{
}

QString const& PVCore::PVFileSerialize::get_path() const
{
	return _path;
}

void PVCore::PVFileSerialize::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.file("data", _path);
}
