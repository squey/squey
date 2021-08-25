/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVCORE_PVSERIALIZEOBJECT_H
#define PVCORE_PVSERIALIZEOBJECT_H

#include <pvkernel/core/PVSerializeArchiveExceptions.h>

#include <cstddef> // for size_t
#include <cstdint> // for uint32_t
#include <memory>  // for __shared_ptr, shared_ptr
#include <string>  // for string
#include <vector>  // for vector, vector<>::iterator

#include <QFile>
#include <QString>
#include <QVariant>

namespace PVCore
{

class PVArgumentList;
class PVSerializeArchive;
class PVSerializeArchiveFixError;

typedef std::shared_ptr<PVSerializeArchive> PVSerializeArchive_p;

/*! \brief Serialization file error
 * Exception that is thrown when a file error has occured.
 */
class PVSerializeObjectFileError : public PVSerializeArchiveError
{
  public:
	explicit PVSerializeObjectFileError(QFile const& file)
	    : PVSerializeArchiveError(QString("Error with file '%1': %2 (%3)")
	                                  .arg(file.fileName())
	                                  .arg(file.errorString())
	                                  .arg(file.error())
	                                  .toStdString())
	{
	}
};

/*! \brief Main serialization object
 *
 * This class is the main helper class used for object serialisation.
 */
class PVSerializeObject
{
	friend class PVSerializeArchive;
	friend class PVSerializeArchiveFixError;

  public:
	typedef std::shared_ptr<PVSerializeObject> p_type;

  protected:
	PVSerializeObject(QString path, PVSerializeArchive* parent_ar);

  public:
	virtual ~PVSerializeObject() = default;

  private:
	/*! \brief Private copy-constructor
	 *  Private copy-constructor, as these objects must always be created by
	 *  PVSerializeArchive::create_object
	 */
	PVSerializeObject(const PVSerializeObject& src);

  public:
	/*! \brief Split reading and writing serialization process
	 *  This can be called by any object being serialized to call different function for the writing
	 * and reading process.
	 */
	template <typename T>
	void split(T& obj)
	{
		if (is_writing()) {
			obj.serialize_write(*this);
		} else {
			obj.serialize_read(*this);
		}
	}

	QString const& get_logical_path() const;
	void set_current_status(std::string const& s);

  public:
	bool is_writing() const;

	/*! \brief Declare a new object to serialize that can be optionally saved, with a description.
	 *  \param[in] name Name of the object to serialize
	 *  \param[in,out] obj Object to load/save
	 *
	 *  This method will create a new PVSerializeObject and call the
	 *  serialize_(read/write) method of obj with this new PVSerializeObject.
	 */
	template <typename T>
	void object(QString const& name, T& obj);

	bool save_log_file() const;

	/*! \brief Declare a list of attributes to load/save in the 'configuration' of the object. T
	 * must be a STL-compliant container. T::value_type must be convertible to and from a QVariant.
	 *  \param[in] name Name of the list of attributes
	 *  \param[in,out] obj List of attributes to save
	 *  These attributes will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void list_attributes(QString const& name, T& obj);

	/*! \brief Declare a list of attributes to load/save in the 'configuration' of the object. T
	 * must be a STL-compliant container. T::value_type must be convertible to and from a QVariant.
	 *  \param[in] name Name of the list of attributes
	 *  \param[in] variant_conv Function that takes a QVaraint and returns an object of type
	 * T::value_type
	 *  \param[in,out] obj List of attributes to save
	 *  These attributes will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T, class F>
	void list_attributes(QString const& name, T& obj, F const& variant_conv);

	/*! \brief Read/save a buffer for this object
	 *  \param[in] name Name of the buffer. This will be used for the underlying filename.
	 *  \param[in,out] buf Original/destination buffer
	 *  \param[in] n Size of the buffer
	 *  \return The number of bytes read/written
	 *  This will read/save the buffer pointed by buf. name is used as the underlying filename.
	 */
	size_t buffer_read(QString const& name, void* buf, size_t n);
	size_t buffer_write(QString const& name, void const* buf, size_t n);

	template <typename T>
	size_t buffer_read(QString const& name, std::vector<T>& buf, size_t n)
	{
		buf.resize(n);
		return buffer_read(name, buf.data(), n * sizeof(T));
	}

	template <typename T>
	size_t buffer_write(QString const& name, std::vector<T> const& buf)
	{
		return buffer_write(name, buf.data(), buf.size() * sizeof(T));
	}

	/*! \brief Include an existing file, given its path.
	 *  \param[in] name Name of this file. This will be used as the underlying destination filename.
	 *  \param[in,out] path Path to the file. When reading the archive, this is set to the extracted
	 * file path.
	 */
	QString file_read(QString const& name);
	void file_write(QString const& name, QString const& path);

	p_type create_object(QString const& name);
	uint32_t get_version() const;

	void attribute_write(QString const& name, QVariant const& obj);
	template <class T>
	T attribute_read(QString const& name)
	{
		return attribute_reader(name).value<T>();
	}

	bool is_repaired_error() const;
	std::string const& get_repaired_value() const;

	void arguments_write(QString const& name, PVArgumentList const& obj);
	void arguments_read(QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);

  private:
	QVariant attribute_reader(QString const& name);
	void list_attributes_write(QString const& name, std::vector<QVariant> const& list);
	void list_attributes_read(QString const& name, std::vector<QVariant>& list);

	template <typename T>
	void call_serialize(T& obj, p_type new_obj)
	{
		obj.serialize(*new_obj, get_version());
	}

  private:
	/*! \brief Parent archive pointer
	 */
	PVSerializeArchive* _parent_ar;

	/*! \brief Logicial path within the archive
	 *  This path is set by PVSerializeArchive::create_object and is independent of the
	 *  final root directory and any operating system rules.
	 *  This can be used to uniquely identify an object.
	 */
	QString _logical_path;
};

typedef PVSerializeObject::p_type PVSerializeObject_p;

template <typename T>
void PVSerializeObject::object(QString const& name, T& obj)
{
	p_type new_obj = create_object(name);
	call_serialize(obj, new_obj);
}

template <class T>
void PVSerializeObject::list_attributes(QString const& name, T& obj)
{
	list_attributes(name, obj,
	                [=](QVariant const& v) { return v.value<typename T::value_type>(); });
}

template <class T, class F>
void PVSerializeObject::list_attributes(QString const& name, T& obj, F const& variant_conv)
{
	if (is_writing()) {
		std::vector<QVariant> list;
		list.reserve(obj.size());
		typename T::const_iterator it;
		for (it = obj.begin(); it != obj.end(); it++) {
			typename T::value_type const& v = *it;
			list.push_back(QVariant(v));
		}
		list_attributes_write(name, list);
	} else {
		std::vector<QVariant> list;
		list_attributes_read(name, list);
		std::vector<QVariant>::iterator it;
		for (it = list.begin(); it != list.end(); it++) {
			obj.push_back(variant_conv(*it));
		}
	}
}
} // namespace PVCore

#endif
