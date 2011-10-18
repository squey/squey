#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

PVCore::PVSerializeObject::PVSerializeObject(QString const& path, PVSerializeArchive_p parent_ar, PVSerializeObject_p parent):
		_parent_ar(parent_ar),
		_parent(parent),
		_logical_path(path),
		_is_optional(false),
		_must_write(true)
{
}

bool PVCore::PVSerializeObject::is_writing() const
{
	return _parent_ar->is_writing();
}

PVCore::PVSerializeObject_p PVCore::PVSerializeObject::create_object(QString const& name, QString const& desc, bool optional)
{
	p_type child = _parent_ar->create_object(name, shared_from_this());
	child->_is_optional = optional;
	child->_desc = (desc.isNull())?name:desc;
	_childs.insert(name, child);
	return child;
}

PVCore::PVSerializeArchive::version_t PVCore::PVSerializeObject::get_version() const
{
	return _parent_ar->get_version();
}

size_t PVCore::PVSerializeObject::buffer(QString const& name, void* buf, size_t n)
{
	return _parent_ar->buffer(*this, name, buf, n);
}

void PVCore::PVSerializeObject::file(QString const& name, QString& path)
{
	_parent_ar->file(*this, name, path);
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

bool PVCore::PVSerializeObject::is_optional() const
{
	return _is_optional;
}

QString const& PVCore::PVSerializeObject::description() const
{
	return _desc;
}

PVCore::PVSerializeObject::list_childs_t const& PVCore::PVSerializeObject::childs() const
{
	return _childs;
}

bool PVCore::PVSerializeObject::must_write() const
{
	assert(is_optional());
	return _must_write;
}

void PVCore::PVSerializeObject::set_write(bool write)
{
	assert(is_optional());
	_must_write = write;
}

const PVCore::PVSerializeObject::p_type PVCore::PVSerializeObject::get_child_by_name(QString const& name) const
{
	return _childs.value(name, p_type());
}

bool PVCore::PVSerializeObject::must_write_child(QString const& name)
{
	return _parent_ar->must_write_object(*this, name);
}

QString const& PVCore::PVSerializeObject::get_logical_path() const
{
	return _logical_path;
}

PVCore::PVSerializeObject::p_type PVCore::PVSerializeObject::parent()
{
	return _parent;
}
