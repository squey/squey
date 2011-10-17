#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

PVCore::PVSerializeObject::PVSerializeObject(QDir const& path, PVSerializeArchive_p parent_ar, PVSerializeObject_p parent):
		_parent_ar(parent_ar),
		_parent(parent),
		_path(path)
{
}

QString PVCore::PVSerializeObject::get_config_path() const
{
	return _path.absoluteFilePath("config.ini");
}

bool PVCore::PVSerializeObject::is_writing() const
{
	return _parent_ar->is_writing();
}

PVCore::PVSerializeObject_p PVCore::PVSerializeObject::create_object(QString const& name)
{
	return _parent_ar->create_object(name, shared_from_this());
}

PVCore::PVSerializeArchive::version_t PVCore::PVSerializeObject::get_version() const
{
	return _parent_ar->get_version();
}

size_t PVCore::PVSerializeObject::buffer(QString const& name, void* buf, size_t n)
{
	QFile buf_file(_path.absoluteFilePath(name));
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
		// QFile::write reports that everything has been written !
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

void PVCore::PVSerializeObject::attribute_write(QString const& name, QVariant const& obj)
{
	_parent_ar->attribute_write(*this, name, obj);
}

void PVCore::PVSerializeObject::attribute_read(QString const& name, QVariant& obj, QVariant const& def)
{
	_parent_ar->attribute_read(*this, name, obj, def);
}

void PVCore::PVSerializeObject::list_attributes_write(QString const& name, std::vector<QVariant> const& list)
{
	_parent_ar->list_attributes_write(*this, name, list);
}

void PVCore::PVSerializeObject::list_attributes_read(QString const& name, std::vector<QVariant>& list)
{
	_parent_ar->list_attributes_read(*this, name, list);
}
