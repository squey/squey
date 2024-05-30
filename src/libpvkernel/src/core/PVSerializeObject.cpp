//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeObject.h>

#include <algorithm> // for move

PVCore::PVSerializeObject::PVSerializeObject(QString path, PVSerializeArchive* parent_ar)
    : _parent_ar(parent_ar), _logical_path(std::move(path))
{
}

bool PVCore::PVSerializeObject::is_writing() const
{
	return _parent_ar->is_writing();
}

PVCore::PVSerializeObject_p PVCore::PVSerializeObject::create_object(QString const& name)
{
	p_type child = _parent_ar->create_object(name, this);
	return child;
}

PVCore::PVSerializeArchive::version_t PVCore::PVSerializeObject::get_version() const
{
	return _parent_ar->get_version();
}

bool PVCore::PVSerializeObject::is_repaired_error() const
{
	return _parent_ar->get_repaired_value().find(get_logical_path().toStdString()) !=
	       _parent_ar->get_repaired_value().end();
}

bool PVCore::PVSerializeObject::save_log_file() const
{
	return _parent_ar->save_log_file();
}

std::string const& PVCore::PVSerializeObject::get_repaired_value() const
{
	return _parent_ar->get_repaired_value().at(get_logical_path().toStdString());
}

size_t PVCore::PVSerializeObject::buffer_read(QString const& name, void* buf, size_t n)
{
	return _parent_ar->buffer_read(*this, name, buf, n);
}

size_t PVCore::PVSerializeObject::buffer_write(QString const& name, void const* buf, size_t n)
{
	return _parent_ar->buffer_write(*this, name, buf, n);
}

QString PVCore::PVSerializeObject::file_read(QString const& name)
{
	return _parent_ar->file_read(*this, name);
}

void PVCore::PVSerializeObject::file_write(QString const& name, QString const& path)
{
	_parent_ar->file_write(*this, name, path);
}

void PVCore::PVSerializeObject::attribute_write(QString const& name, QVariant const& obj)
{
	_parent_ar->attribute_write(*this, name, obj);
}

QVariant PVCore::PVSerializeObject::attribute_reader(QString const& name)
{
	return _parent_ar->attribute_read(*this, name);
}

void PVCore::PVSerializeObject::list_attributes_write(QString const& name,
                                                      std::vector<QVariant> const& list)
{
	_parent_ar->list_attributes_write(*this, name, list);
}

void PVCore::PVSerializeObject::list_attributes_read(QString const& name,
                                                     std::vector<QVariant>& list)
{
	_parent_ar->list_attributes_read(*this, name, list);
}

void PVCore::PVSerializeObject::arguments_write(QString const& name, PVArgumentList const& obj)
{
	_parent_ar->hash_arguments_write(*this, name, obj);
}

void PVCore::PVSerializeObject::arguments_read(QString const& name,
                                               PVArgumentList& obj,
                                               PVArgumentList const& def_args)
{
	_parent_ar->hash_arguments_read(*this, name, obj, def_args);
}

QString const& PVCore::PVSerializeObject::get_logical_path() const
{
	return _logical_path;
}

void PVCore::PVSerializeObject::set_current_status(std::string const& s)
{
	_parent_ar->set_current_status(s);
}
