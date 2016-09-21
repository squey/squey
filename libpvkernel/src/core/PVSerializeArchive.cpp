/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeObject.h>

PVCore::PVSerializeArchive::PVSerializeArchive(version_t version, bool save_log_file)
    : _version(version), _is_opened(false), _save_log_files(save_log_file)
{
}

PVCore::PVSerializeArchive::PVSerializeArchive(QString const& dir,
                                               archive_mode mode,
                                               version_t version,
                                               bool save_log_file)
    : _version(version), _is_opened(false), _save_log_files(save_log_file)
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
				throw PVSerializeArchiveError("Unable to remove directory '" + dir.toStdString() +
				                              "'.");
			}
		}
		if (!QDir::root().mkdir(dir)) {
			throw PVSerializeArchiveError("Unable to create directory '" + dir.toStdString() +
			                              "'.");
		}
	} else {
		if (!dir_.exists()) {
			throw PVSerializeArchiveError("Unable to find directory '" + dir.toStdString() + "'.");
		}
	}

	_is_opened = true;
}

PVCore::PVSerializeArchive::~PVSerializeArchive()
{
	if (_is_opened) {
		for (auto it = _objs_attributes.constBegin(); it != _objs_attributes.constEnd(); it++) {
			delete it.value();
		}
		_root_obj.reset();
	}
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::allocate_object(QString const& name,
                                                                        PVSerializeObject* parent)
{
	QString new_path;
	if (parent) {
		new_path = parent->get_logical_path() + QString("/") + name;
	} else {
		new_path = "";
	}

	PVSerializeObject_p ret(new PVSerializeObject(new_path, this));
	_objects.insert(ret->get_logical_path(), ret);
	return ret;
}

void PVCore::PVSerializeArchive::init()
{
	_root_obj = allocate_object(_root_dir, nullptr);
	create_attributes(*_root_obj);
	// Version special attribute
	if (is_writing()) {
		_root_obj->attribute_write("version", _version);
	} else {
		_version = _root_obj->attribute_read<version_t>("version");
	}
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::create_object(QString const& name,
                                                                      PVSerializeObject* parent)
{
	PVSerializeObject_p ret(allocate_object(name, parent));
	QDir new_path = get_dir_for_object(*ret);
	if (is_writing()) {
		if (!QDir::root().mkdir(new_path.path())) {
			throw PVSerializeArchiveError("Unable to create directory '" +
			                              new_path.canonicalPath().toStdString() + "'.");
		}
	}
	if (!new_path.exists()) {
		throw PVSerializeArchiveErrorNoObject("Unable to change into directory '" +
		                                      new_path.canonicalPath().toStdString() + "'.");
	}
	create_attributes(*ret);
	return ret;
}

void PVCore::PVSerializeArchive::create_attributes(PVSerializeObject const& so)
{
	_objs_attributes.insert(get_object_config_path(so),
	                        new QSettings(get_object_config_path(so), QSettings::IniFormat));
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

void PVCore::PVSerializeArchive::attribute_write(PVSerializeObject const& so,
                                                 QString const& name,
                                                 QVariant const& obj)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	settings->setValue(name, obj);
}

QVariant PVCore::PVSerializeArchive::attribute_read(PVSerializeObject& so, QString const& name)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	return settings->value(name);
}

void PVCore::PVSerializeArchive::list_attributes_write(PVSerializeObject const& so,
                                                       QString const& name,
                                                       std::vector<QVariant> const& obj)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	settings->beginWriteArray(name);
	int idx = 0;
	for (QVariant const& v : obj) {
		settings->setArrayIndex(idx++);
		settings->setValue("value", v);
	}
	settings->endArray();
}

void PVCore::PVSerializeArchive::list_attributes_read(PVSerializeObject const& so,
                                                      QString const& name,
                                                      std::vector<QVariant>& obj)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	int size = settings->beginReadArray(name);
	obj.clear();
	obj.reserve(size);
	for (int i = 0; i < size; i++) {
		settings->setArrayIndex(i);
		obj.push_back(settings->value("value"));
	}
	settings->endArray();
}

void PVCore::PVSerializeArchive::hash_arguments_write(PVSerializeObject const& so,
                                                      QString const& name,
                                                      PVArgumentList const& obj)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	PVArgumentList_to_QSettings(obj, *settings, name);
}

void PVCore::PVSerializeArchive::hash_arguments_read(PVSerializeObject const& so,
                                                     QString const& name,
                                                     PVArgumentList& obj,
                                                     PVArgumentList const& def_args)
{
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	obj.clear();
	obj = QSettings_to_PVArgumentList(*settings, def_args, name);
}

size_t PVCore::PVSerializeArchive::buffer_read(PVSerializeObject const& so,
                                               QString const& name,
                                               void* buf,
                                               size_t n)
{
	QFile buf_file(get_dir_for_object(so).absoluteFilePath(name));
	if (!buf_file.open(QIODevice::ReadOnly)) {
		throw PVSerializeObjectFileError(buf_file);
	}
	qint64 ret = buf_file.read((char*)buf, n);
	if (ret == -1) {
		throw PVSerializeObjectFileError(buf_file);
	}
	return ret;
}

size_t PVCore::PVSerializeArchive::buffer_write(PVSerializeObject const& so,
                                                QString const& name,
                                                void const* buf,
                                                size_t n)
{
	QFile buf_file(get_dir_for_object(so).absoluteFilePath(name));

	if (!buf_file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		throw PVSerializeObjectFileError(buf_file);
	}
	qint64 ret = buf_file.write((const char*)buf, n);
	if (ret == -1) {
		throw PVSerializeObjectFileError(buf_file);
	}
	// Flush the internal buffers and check that it works.
	// If we don' t do that, we may have an empty file even if
	// QFile::write reports that everything was written !
	if (!buf_file.flush()) {
		throw PVSerializeObjectFileError(buf_file);
	}

	return ret;
}

QDir PVCore::PVSerializeArchive::get_dir_for_object(PVSerializeObject const& so) const
{
	QString lp = so.get_logical_path();
	if (lp.startsWith(QChar('/'))) {
		lp = lp.mid(1);
	}
	return QDir(QDir(_root_dir).absoluteFilePath(lp));
}

QString PVCore::PVSerializeArchive::get_object_config_path(PVSerializeObject const& so) const
{
	QDir dir = get_dir_for_object(so);
	return dir.absoluteFilePath("config.ini");
}

QString PVCore::PVSerializeArchive::file_read(PVSerializeObject const& so, QString const& name)
{
	QDir dir = get_dir_for_object(so);
	QString ar_file(dir.absoluteFilePath(name));
	if (!dir.exists(name)) {
		throw PVSerializeArchiveError("File '" + name.toStdString() + "' within '" +
		                              so.get_logical_path().toStdString() + "' does not exist.");
	}
	return ar_file;
}

void PVCore::PVSerializeArchive::file_write(PVSerializeObject const& so,
                                            QString const& name,
                                            QString const& path)
{
	QDir dir = get_dir_for_object(so);
	QString ar_file(dir.absoluteFilePath(name));
	QFile org_file(path);
	if (!org_file.copy(ar_file)) {
		throw PVSerializeObjectFileError(org_file);
	}
}
