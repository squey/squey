/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEOBJECT_H
#define PVCORE_PVSERIALIZEOBJECT_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVTypeInfo.h>
#include <pvkernel/core/PVTypeTraits.h>

#include <QDir>
#include <QSettings>

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

namespace PVCore
{

class PVSerializeArchive;
class PVSerializeArchiveFixError;

typedef std::shared_ptr<PVSerializeArchive> PVSerializeArchive_p;

/*! \brief Serialization file error
 * Exception that is thrown when a file error has occured.
 */
class PVSerializeObjectFileError : public PVSerializeArchiveError
{
  public:
	PVSerializeObjectFileError(QFile const& file)
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
	friend class PVSerializeArchiveFixAttribute;

  public:
	typedef std::shared_ptr<PVSerializeObject> p_type;
	typedef QHash<QString, p_type> list_childs_t;

  protected:
	PVSerializeObject(QString path,
	                  PVSerializeArchive* parent_ar,
	                  PVSerializeObject* parent = nullptr);

  public:
	virtual ~PVSerializeObject() {}

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

	bool is_writing() const;
	bool is_optional() const;
	bool must_write() const;
	bool visible() const;
	void set_write(bool write);
	QString const& description() const;
	list_childs_t const& childs() const;
	list_childs_t const& visible_childs() const;
	const p_type get_child_by_name(QString const& name) const;
	QString const& get_logical_path() const;
	PVSerializeObject* parent();
	PVTypeInfo const& bound_obj_type() const { return _bound_obj_type; }
	bool has_repairable_errors() const;

  public:
	/*! \brief Declare a new object to serialize that can be optionally saved, with a description.
	 *  \param[in] name Name of the object to serialize
	 *  \param[in,out] obj Object to load/save
	 *  \param[in] optional Can this object be optionally saved
	 *  \param[in] desc Description of the object (for usage in a GUI for instance)
	 *  \param[in] visible Whether or not this object will be exposed by
	 *PVInspector::PVSerializeOptionsModel (for instance)
	 *  \param[in] def_v If not nullptr, pointer to a default value to use when creating the object
	 *(will call T's copy constructor, which must exists).
	 *  \param[in] def_option If optional is equal to true, defines the default optional state of
	 *this object (true = object will be included).
	 *  \return If this object can be optionaly saved and an archive is being read, it returns true
	 *if the object exists in this archive and was successfully read.
	 *
	 *  This method will create a new PVSerializeObject and call the
	 *  serialize_(read/write) method of obj with this new PVSerializeObject.
	 */
	template <typename T>
	bool object(QString const& name,
	            T& obj,
	            QString const& desc = QString(),
	            bool optional = false,
	            typename PVTypeTraits::remove_shared_ptr<T>::type const* def_v = nullptr,
	            bool visible = true,
	            bool def_option = true,
	            p_type* used_so = nullptr);

	/*! \brief Declare a QHash to serialize. V must be serializable.
	 *  \param[in]     name Name of the QHash to serialize.
	 *  \param[in,out] obj  QHash input/output object to serialize.
	 *  This method declare a QHash of object to serialize.
	 *  K has to be convertible to a QVariant.
	 *  This will be represented in this form:
	 *    hash_name
	 *    |
	 *    -- key of first object
	 *       |
	 *       -- serialization of first object
	 *    -- key of second object
	 *       |
	 *       -- serialization of the second object
	 *    ...
	 */
	template <typename K, typename V>
	void hash(QHash<K, V> const& obj);

	void arguments(QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);

	/*! \brief Declare an attribute to load/save in the 'configuration' of the object. T must be
	 * convertible to and from a QVariant
	 *  \param[in]     name Name of the attribute
	 *  \param[in,out] obj  Attribute to save
	 *  \param[in]     def  When reading, default value for obj
	 *  This attribute will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void attribute(QString const& name, T& obj, T const& def = T());

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
	size_t buffer(QString const& name, void* buf, size_t n);

	template <typename T>
	size_t buffer(QString const& name, std::vector<T>& buf, size_t n)
	{
		if (not is_writing()) {
			buf.resize(n);
		}

		return buffer(name, buf.data(), n * sizeof(T));
	}

	/*! \brief Include an existing file, given its path.
	 *  \param[in] name Name of this file. This will be used as the underlying destination filename.
	 *  \param[in,out] path Path to the file. When reading the archive, this is set to the extracted
	 * file path.
	 */
	void file(QString const& name, QString& path);

	/*! \brief Declare an error in the archive (while reading it) that can be repaired by further
	 * user actions.
	 */
	void repairable_error(std::shared_ptr<PVSerializeArchiveFixError> const& error);

  protected:
	void error_fixed(PVSerializeArchiveFixError* error);
	void fix_attribute(QString const& name, QVariant const& obj);
	inline const void* bound_obj() const { return _bound_obj; }

  public:
	template <class T>
	inline void set_bound_obj(T& t)
	{
		_bound_obj = &t;
		_bound_obj_type = typeid(T);
	}

	p_type create_object(QString const& name,
	                     QString const& desc = QString(),
	                     bool optional = false,
	                     bool visible = true,
	                     bool def_option = true);
	uint32_t get_version() const;

  private:
	void attribute_write(QString const& name, QVariant const& obj);
	void attribute_read(QString const& name, QVariant& obj, QVariant const& def);
	void list_attributes_write(QString const& name, std::vector<QVariant> const& list);
	void list_attributes_read(QString const& name, std::vector<QVariant>& list);
	void hash_arguments_write(QString const& name, PVArgumentList const& obj);
	void
	hash_arguments_read(QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);
	bool must_write_child(QString const& name);

	template <typename T>
	void call_serialize(T& obj, p_type new_obj, T const* /*def_v*/)
	{
		obj.serialize(*new_obj, get_version());
		new_obj->set_bound_obj(obj);
	}

  private:
	/*! \brief Parent archive pointer
	 */
	PVSerializeArchive* _parent_ar;

	/*! \brief Parent serialization object
	 */
	PVSerializeObject* _parent;

	/*! \brief Logicial path within the archive
	 *  This path is set by PVSerializeArchive::create_object and is independent of the
	 *  final root directory and any operating system rules.
	 *  This can be used to uniquely identify an object.
	 */
	QString _logical_path;

	/*! \brief Specifies whether this object is optional or not. Set by create_object.
	 */
	bool _is_optional;

	/*! \brief Specifies a description of this object.
	 */
	QString _desc;

	/*! \brief If this object is optional, specifies whether this object must be written or not.
	 */
	bool _must_write;

	/*! \brief Hash of the children of this object. The key is their name.
	 */
	list_childs_t _childs;

	/*! \brief Hash of the visible children of this object. The key is their name.
	 */
	list_childs_t _visible_childs;

	/*! \brief Whether or not this object is exposed to the user (for options)
	 */
	bool _visible;

  public:
	/*! \brief If relevant, represents a pointer to the object that has been serialized
	 */
	void* _bound_obj;
	PVTypeInfo _bound_obj_type;
};

typedef PVSerializeObject::p_type PVSerializeObject_p;

template <typename T>
bool PVSerializeObject::object(QString const& name,
                               T& obj,
                               QString const& desc,
                               bool optional,
                               typename PVTypeTraits::remove_shared_ptr<T>::type const* def_v,
                               bool visible,
                               bool def_option,
                               p_type* used_so)
{
	if (optional && is_writing()) {
		if (!must_write_child(name)) {
			return false;
		}
	}
	p_type new_obj;
	try {
		new_obj = create_object(name, desc, optional, visible, def_option);
	} catch (PVSerializeArchiveErrorNoObject& e) {
		if (!optional && !is_writing()) {
			throw e;
		}
		return false;
	}
	if (used_so) {
		*used_so = new_obj;
	}
	call_serialize(obj, new_obj, def_v);
	return true;
}

template <typename K, typename V>
void PVSerializeObject::hash(QHash<K, V> const& obj)
{
	if (is_writing()) {
		for (auto it = obj.begin(); it != obj.end(); it++) {
			PVSerializeObject_p new_obj = create_object(it.key());
			it.value().serialize(*new_obj);
		}
	}
}

template <class T>
void PVSerializeObject::attribute(QString const& name, T& obj, T const& def)
{
	if (is_writing()) {
		attribute_write(name, QVariant(obj));
	} else {
		QVariant v;
		attribute_read(name, v, QVariant(def));
		obj = v.value<T>();
	}
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
}

// Conveniance macros
#define PVSERIALIZEOBJECT_SPLIT                                                                    \
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)     \
	{                                                                                              \
		so.split(*this);                                                                           \
	}

#endif
