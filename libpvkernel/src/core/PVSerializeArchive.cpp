/**
 * \file PVSerializeArchive.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
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
#ifdef CUSTOMER_CAPABILITY_SAVE
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
#endif
}

PVCore::PVSerializeArchive::~PVSerializeArchive()
{
	if (_is_opened) {
		finish();
	}
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::allocate_object(QString const& name, PVSerializeObject* parent)
{
	QString new_path;
	if (parent) {
		new_path = parent->get_logical_path() + QString("/") + name;
	}
	else {
		new_path = "";
	}

	PVSerializeObject_p ret(new PVSerializeObject(new_path, this, parent));
	_objects.insert(get_object_logical_path(*ret), ret);
	return ret;
}

void PVCore::PVSerializeArchive::init()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	_root_obj = allocate_object(_root_dir, NULL);
	create_attributes(*_root_obj);
	// Version special attribute
	_root_obj->attribute(QString("version"), _version, (version_t) 0);
#endif
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::create_object(QString const& name, PVSerializeObject* parent)
{
	PVSerializeObject_p ret(allocate_object(name, parent));
	QDir new_path = get_dir_for_object(*ret);
	if (is_writing()) {
		if (!QDir::root().mkdir(new_path.path())) {
			throw PVSerializeArchiveError(QString("Unable to create directory '%1'.").arg(new_path.canonicalPath()));
		}
	}
	if (!new_path.exists()) {
		throw PVSerializeArchiveErrorNoObject(name, QString("Unable to change into directory '%1'.").arg(new_path.canonicalPath()));
	}
	create_attributes(*ret);
	return ret;
}

void PVCore::PVSerializeArchive::create_attributes(PVSerializeObject const& so)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	_objs_attributes.insert(get_object_config_path(so), new QSettings(get_object_config_path(so), QSettings::IniFormat));
#endif
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
#ifdef CUSTOMER_CAPABILITY_SAVE
	QHash<QString, QSettings*>::const_iterator it;
	for (it = _objs_attributes.constBegin(); it != _objs_attributes.constEnd(); it++) {
		delete it.value();
	}
	_root_obj.reset();
	_is_opened = false;
#endif
}

void PVCore::PVSerializeArchive::attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	settings->setValue(name, obj);
#endif
}

void PVCore::PVSerializeArchive::attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	obj = settings->value(name, def);
#endif
}

void PVCore::PVSerializeArchive::list_attributes_write(PVSerializeObject const& so, QString const& name, std::vector<QVariant> const& obj)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	std::vector<QVariant>::const_iterator it;
	settings->beginWriteArray(name);
	int idx = 0;
	for (it = obj.begin(); it != obj.end(); it++) {
			settings->setArrayIndex(idx);
			settings->setValue("value", *it);
			idx++;
	}
	settings->endArray();
#endif
}

void PVCore::PVSerializeArchive::list_attributes_read(PVSerializeObject const& so, QString const& name, std::vector<QVariant>& obj)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	int size = settings->beginReadArray(name);
	obj.clear();
	obj.reserve(size);
	for (int i = 0; i < size; i++) {
		settings->setArrayIndex(i);
		obj.push_back(settings->value("value"));
	}
	settings->endArray();
#endif
}

void PVCore::PVSerializeArchive::hash_arguments_write(PVSerializeObject const& so, QString const& name, PVArgumentList const& obj)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
//	PVArgumentList::const_iterator it;
//	settings->beginGroup(name);
//	for (it = obj.begin(); it != obj.end(); it++) {
//		   settings->setValue(it.key(), PVArgument_to_QString(it.value()));
//	}
//	settings->endGroup();
	PVArgumentList_to_QSettings(obj, *settings, name);
#endif
}

void PVCore::PVSerializeArchive::hash_arguments_read(PVSerializeObject const& so, QString const& name, PVArgumentList& obj, PVArgumentList const& def_args)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QSettings* settings = _objs_attributes.value(get_object_config_path(so));
	obj.clear();
//	QStringList keys = settings->childKeys();
//	for (int i = 0; i < keys.size(); i++) {
//		QString const& key = keys.at(i);
//		if (def_args.contains(key)) {
//			obj[key] = QString_to_PVArgument(settings->value(key).toString(), def_args[key]);
//		}
//	}
	obj = QSettings_to_PVArgumentList(*settings, def_args, name);
#endif
}

size_t PVCore::PVSerializeArchive::buffer(PVSerializeObject const& so, QString const& name, void* buf, size_t n)
{
	QFile buf_file(get_dir_for_object(so).absoluteFilePath(name));
	if (is_writing()) {
		if (!buf_file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			throw PVSerializeObjectFileError(buf_file);
		}
		qint64 ret = buf_file.write((const char*) buf, n);
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
	else {
		if (!buf_file.open(QIODevice::ReadOnly)) {
			throw PVSerializeObjectFileError(buf_file);
		}
		qint64 ret = buf_file.read((char*) buf, n);
		if (ret == -1) {
			throw PVSerializeObjectFileError(buf_file);
		}
		return ret;
	}
}

bool PVCore::PVSerializeArchive::must_write_object(PVSerializeObject const& parent, QString const& child)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	assert(_options.get() != this);
	if (!_options) {
		return true;
	}
	return _options->must_write(parent, child);
#else
	return false;
#endif
}

QDir PVCore::PVSerializeArchive::get_dir_for_object(PVSerializeObject const& so) const
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QString lp = so.get_logical_path();
	if (lp.startsWith(QChar('/'))) {
		lp = lp.mid(1);
	}
	return QDir(QDir(_root_dir).absoluteFilePath(lp));
#else
	return QDir();
#endif
}

QString PVCore::PVSerializeArchive::get_object_config_path(PVSerializeObject const& so) const
{
	QDir dir = get_dir_for_object(so);
	return dir.absoluteFilePath("config.ini");
}

void PVCore::PVSerializeArchive::file(PVSerializeObject const& so, QString const& name, QString& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QDir dir = get_dir_for_object(so);
	QString ar_file(dir.absoluteFilePath(name));
	if (is_writing()) {
		QFile org_file(path);
		if (!org_file.copy(ar_file)) {
			throw PVSerializeObjectFileError(org_file);
		}
	}
	else {
		if (!dir.exists(name)) {
			throw PVSerializeArchiveError(QString("File '%1' within '%2' does not exist.").arg(name).arg(so.get_logical_path()));
		}
		path = ar_file;
	}
#endif
}

PVCore::PVSerializeObject_p PVCore::PVSerializeArchive::get_object_by_path(QString const& path) const
{
	assert(_objects.contains(path));
	return _objects[path];
}

void PVCore::PVSerializeArchive::repairable_error(boost::shared_ptr<PVSerializeArchiveFixError> const& error)
{
	_repairable_errors.push_back(error);
}

void PVCore::PVSerializeArchive::error_fixed(PVSerializeArchiveFixError* error)
{
	list_errors_t::iterator it;
	for (it = _repairable_errors.begin(); it != _repairable_errors.end(); it++) {
		if (it->get() == error) {
			_repairable_errors.erase(it);
			return;
		}
	}
}
