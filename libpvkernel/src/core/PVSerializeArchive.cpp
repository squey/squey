#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeObject.h>

PVCore::PVSerializeArchive::PVSerializeArchive(QString const& dir, archive_mode mode, version_t version):
	_mode(mode),
	_root_dir(dir),
	_version(version)
{
	QDir dir_(dir);
	if (mode == write) {
		if (dir_.exists()) {
			if (!PVDirectory::remove_rec(dir)) {
				throw PVSerializeArchiveError(QString("Unable to remove directory '%1'.").arg(dir));
			}
		}
		if (!QDir::root().mkdir(dir)) {
			throw PVSerializeArchiveError(QString("Unable to create directory '%1'.").arg(dir));
		}
	}
	else {
		if (!dir_.exists()) {
				throw PVSerializeArchiveError(QString("Unable to find directory '%1'.").arg(dir));
		}
	}
}

void PVCore::PVSerializeArchive::init()
{
	_root_obj = PVSerializeObject_p(new PVSerializeObject(_root_dir, shared_from_this()));

	// Version special attribute
	_root_obj->attribute(QString("version"), _version, (version_t) 0);
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::create_object(QString const& name, PVSerializeObject_p parent)
{
	QDir new_path(parent->get_path());
	if (is_writing()) {
		if (!new_path.mkdir(name)) {
			throw PVSerializeArchiveError(QString("Unable to create directory '%1' within '%2'.").arg(name).arg(new_path.absolutePath()));
		}
	}
	if (!new_path.cd(name)) {
		throw PVSerializeArchiveError(QString("Unable to change into directory '%1' within '%2'.").arg(name).arg(new_path.absolutePath()));
	}
	PVSerializeObject_p ret(new PVSerializeObject(new_path, shared_from_this(), parent));
	return ret;
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::get_root()
{
	if (!_root_obj) {
		init();
	}
	return _root_obj;
}

PVCore::PVSerializeArchive::version_t PVCore::PVSerializeArchive::get_version()
{
	return _version;
}
