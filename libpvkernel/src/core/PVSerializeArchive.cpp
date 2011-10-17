#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeObject.h>

PVCore::PVSerializeArchive::PVSerializeArchive(version_t version):
	_version(version),
	_is_opened(false)
{
}


PVCore::PVSerializeArchive::PVSerializeArchive(QString const& dir, archive_mode mode, version_t version):
	_version(version),
	_is_opened(false)
{
	open(dir, mode);
}

void PVCore::PVSerializeArchive::open(QString const& dir, archive_mode mode)
{
	_mode = mode;
	_root_dir = dir;

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

	_is_opened = true;
}

PVCore::PVSerializeArchive::~PVSerializeArchive()
{
	if (_is_opened) {
		finish();
	}
}

void PVCore::PVSerializeArchive::init()
{
	_root_obj = PVSerializeObject_p(new PVSerializeObject(_root_dir, shared_from_this()));
	create_attributes(*_root_obj);

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
	create_attributes(*ret);
	return ret;
}

void PVCore::PVSerializeArchive::create_attributes(PVSerializeObject const& so)
{
	_objs_attributes.insert(so.get_config_path(), new QSettings(so.get_config_path(), QSettings::IniFormat));
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::get_root()
{
	if (!_root_obj) {
		init();
	}
	return _root_obj;
}

PVCore::PVSerializeArchive::version_t PVCore::PVSerializeArchive::get_version() const
{
	return _version;
}

void PVCore::PVSerializeArchive::finish()
{
	QHash<QString, QSettings*>::const_iterator it;
	for (it = _objs_attributes.constBegin(); it != _objs_attributes.constEnd(); it++) {
		delete it.value();
	}
	_root_obj.reset();
	_is_opened = false;
}

void PVCore::PVSerializeArchive::attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj)
{
	QSettings* settings = _objs_attributes.value(so.get_config_path());
	settings->setValue(name, obj);
}

void PVCore::PVSerializeArchive::attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def)
{
	QSettings* settings = _objs_attributes.value(so.get_config_path());
	obj = settings->value(name, def);
}

void PVCore::PVSerializeArchive::list_attributes_write(PVSerializeObject const& so, QString const& name, std::vector<QVariant> const& obj)
{
	QSettings* settings = _objs_attributes.value(so.get_config_path());
	std::vector<QVariant>::const_iterator it;
	settings->beginWriteArray(name);
	int idx = 0;
	for (it = obj.begin(); it != obj.end(); it++) {
			settings->setArrayIndex(idx);
			settings->setValue("value", *it);
			idx++;
	}
	settings->endArray();
}

void PVCore::PVSerializeArchive::list_attributes_read(PVSerializeObject const& so, QString const& name, std::vector<QVariant>& obj)
{
	QSettings* settings = _objs_attributes.value(so.get_config_path());
	int size = settings->beginReadArray(name);
	obj.clear();
	obj.reserve(size);
	for (int i = 0; i < size; i++) {
		settings->setArrayIndex(i);
		obj.push_back(settings->value("value"));
	}
	settings->endArray();
}
