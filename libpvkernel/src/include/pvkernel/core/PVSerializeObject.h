/**
 * \file PVSerializeObject.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVSERIALIZEOBJECT_H
#define PVCORE_PVSERIALIZEOBJECT_H

#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVDataTreeAutoShared.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVTypeTraits.h>
#include <pvkernel/core/PVTypeInfo.h>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits.hpp>

#include <QDir>
#include <QSettings>

#include <vector>

namespace PVCore {

class PVSerializeArchive;
class PVSerializeArchiveFixError;

typedef boost::shared_ptr<PVSerializeArchive> PVSerializeArchive_p;

/*! \brief Serialization file error
 * Exception that is thrown when a file error has occured.
 */
class LibKernelDecl PVSerializeObjectFileError: public PVSerializeArchiveError 
{
public:
	PVSerializeObjectFileError(QFile const& file):
		PVSerializeArchiveError(QString("Error with file '%1': %2 (%3)").arg(file.fileName()).arg(file.errorString()).arg(file.error()))
	{ }
};

/*! \brief Main serialization object
 *
 * This class is the main helper class used for object serialisation.
 */
class LibKernelDecl PVSerializeObject: public boost::enable_shared_from_this<PVSerializeObject>
{
	friend class PVSerializeArchive;
	friend class PVSerializeArchiveFixError;
	friend class PVSerializeArchiveFixAttribute;

public:
	typedef boost::shared_ptr<PVSerializeObject> p_type;
	typedef QHash<QString, p_type> list_childs_t;

protected:
	PVSerializeObject(QString const& logical_path, PVSerializeArchive* parent_ar, PVSerializeObject* parent = NULL);

public:
	virtual ~PVSerializeObject() { }

private:
	/*! \brief Private copy-constructor
	 *  Private copy-constructor, as these objects must always be created by
	 *  PVSerializeArchive::create_object
	 */
	PVSerializeObject(const PVSerializeObject& src);

public:
	/*! \brief Split reading and writing serialization process
	 *  This can be called by any object being serialized to call different function for the writing and reading process.
	 */
	template <typename T>
	void split(T& obj)
	{
		if (is_writing()) {
			obj.serialize_write(*this);
		}
		else {
			obj.serialize_read(*this, get_version());
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

	template <class T>
	QString get_child_path(T const* obj) const
	{
		list_childs_t const& cen = childs();
		list_childs_t::const_iterator it;
		for (it = cen.begin(); it != cen.end(); it++) {
			p_type const& c = it.value();
			if (c->bound_obj_as<T>() == obj) {
				return c->get_logical_path();
			}
		}
		return QString();
	}

	template <class T>
	QString get_child_path(boost::shared_ptr<T> const& obj) const
	{
		list_childs_t const& cen = childs();
		list_childs_t::const_iterator it;
		for (it = cen.begin(); it != cen.end(); it++) {
			p_type const& c = it.value();
			if (c->bound_obj_as<T>() == obj.get()) {
				return c->get_logical_path();
			}
		}
		return QString();
	}

	template <class T>
	QString get_child_path(PVDataTreeAutoShared<T> const& obj) const
	{
		list_childs_t const& cen = childs();
		list_childs_t::const_iterator it;
		for (it = cen.begin(); it != cen.end(); it++) {
			p_type const& c = it.value();
			if (c->bound_obj_as<T> == obj.get()) {
				return c->get_logical_path();
			}
		}
		return QString();
	}

	template <class T>
	QString get_child_path(PVSharedPtr<T> const& obj) const
	{
		list_childs_t const& cen = childs();
		list_childs_t::const_iterator it;
		for (it = cen.begin(); it != cen.end(); it++) {
			p_type const& c = it.value();
			if (c->bound_obj_as<T>() == obj.get()) {
				return c->get_logical_path();
			}
		}
		return QString();
	}

	bool object_exists_by_path(QString const& path) const;
	p_type get_object_by_path(QString const& path) const;

	template <class T>
	inline T* bound_obj_as() const
	{
		if (bound_obj_type() != typeid(T) || !_bound_obj) {
			return NULL;
		}
		return (T*) _bound_obj;
	}

public:
	/*! \brief Declare a new object to serialize that can be optionally saved, with a description.
	 *  \param[in] name Name of the object to serialize
	 *  \param[in,out] obj Object to load/save
	 *  \param[in] optional Can this object be optionally saved
	 *  \param[in] desc Description of the object (for usage in a GUI for instance)
	 *  \param[in] visible Whether or not this object will be exposed by PVInspector::PVSerializeOptionsModel (for instance)
	 *  \param[in] def_v If not NULL, pointer to a default value to use when creating the object (will call T's copy constructor, which must exists).
	 *  \param[in] def_option If optional is equal to true, defines the default optional state of this object (true = object will be included).
	 *  \return If this object can be optionaly saved and an archive is being read, it returns true if the object exists in this archive and was successfully read.
	 *
	 *  This method will create a new PVSerializeObject and call the
	 *  serialize_(read/write) method of obj with this new PVSerializeObject.
	 */
	template <typename T>
	bool object(QString const& name, T& obj, QString const& desc = QString(), bool optional = false, typename PVTypeTraits::remove_shared_ptr<T>::type const* def_v = NULL, bool visible = true, bool def_option = true, p_type* used_so = NULL);

	/*! \brief Declare a list to serialize. T must be an STL-compliant container. T::value_type must be serializable.
	 *  \param[in] name Name of the list to serialize
	 *  \param[in] obj List object to serialize
	 *  \param[in] def_v Default value to use when an object has to be created (when using shared_ptr for instance)
	 *  \param[in] descriptions Description to use for the objects inside the list
	 *  \param[in] visible Whether this list object will be visible or not
	 *  \param[in] elts_optional Whether the lements of this list will be optional
	 *  \param[in] desc Optional description of this list. If an empty QString is given, no description will be associated.
	 *  This method declare a list of object to serialize.
	 *  This will be represented in this form:
	 *    list_name
	 *    |
	 *    -- 0
	 *       |
	 *       -- serialization of first object
	 *    -- 1
	 *       |
	 *       -- serialization of the second object
	 *    ...
	 */
	template <typename T>
	p_type list(QString const& name, T& obj, QString const& desc = QString(), typename PVTypeTraits::remove_shared_ptr<typename T::value_type>::type const* def_v = NULL, QStringList const& descriptions = QStringList(), bool visible = true, bool elts_optional = false);

	// C++ 0x is coming, but still too experimental and we may have trouble with MSVC...
	//template <typename T, typename V = typename T::value_type>
	template <typename T, typename V>
	p_type list(QString const& name, T& obj, QString const& desc = QString(), typename PVTypeTraits::remove_shared_ptr<V>::type const* def_v = NULL, QStringList const& descriptions = QStringList(), bool visible = true, bool elts_optional = false);

	template <typename F>
	p_type list_read(F const& func, QString const& name, QString const& desc = QString(), bool visible = true, bool elts_optional = false);

	/*! \brief Declare a list to serialize by making references to objects that has already been serialized.
	 *  \param[in] name Name of the list to serialize
	 *  \param[in] obj List to serialize
	 *  \param[in] ref_so Serialized object returned by a previous call to PVSerializeObject::list
	 *
	 *  This method declare a list of object to serialize by making references to objects that has already been serialized.
	 *  Every elements of T must have already been serialized inside ref_so.
	 *  It emulates a 1-to-n relationship.
	 *  T must be an STL-compliant container. T::value_type must be serializable.
	 */
	template <typename T>
	void list_ref(QString const& name, T& obj, p_type ref_so);

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
	void hash(QString const& name, QHash<K,V>& obj);

	void arguments(QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);

	/*! \brief Declare an attribute to load/save in the 'configuration' of the object. T must be convertible to and from a QVariant
	 *  \param[in]     name Name of the attribute
	 *  \param[in,out] obj  Attribute to save
	 *  \param[in]     def  When reading, default value for obj 
	 *  This attribute will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void attribute(QString const& name, T& obj, T const& def = T());

	/*! \brief Declare a list of attributes to load/save in the 'configuration' of the object. T must be a STL-compliant container. T::value_type must be convertible to and from a QVariant.
	 *  \param[in] name Name of the list of attributes
	 *  \param[in,out] obj List of attributes to save
	 *  These attributes will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void list_attributes(QString const& name, T& obj);

	/*! \brief Declare a list of attributes to load/save in the 'configuration' of the object. T must be a STL-compliant container. T::value_type must be convertible to and from a QVariant.
	 *  \param[in] name Name of the list of attributes
	 *  \param[in] variant_conv Function that takes a QVaraint and returns an object of type T::value_type
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

	/*! \brief Read a buffer for this object, by just providing the path to its underlying filename.
	 *  \param[in] name Name of the buffer. This will be used for the underlying filename.
	 *  \param[in,out] path Path to the file. When reading the archive, this is set to the extracted file path.
	 *  \return false is the archive is being read, true otherwise.
	 *  This method can only be used when the archive is being read !
	 */
	bool buffer_path(QString const& name, QString& path);

	/*! \brief Checks whether a buffer exists or not when reading an archive
	 *  \param[in] name Name of the buffer. This will be used for the underlying filename.
	 *  \return true if the buffer exists, false otherwise (or if the archive is being written)
	 */
	bool buffer_exists(QString const& name);

	/*! \brief Include an existing file, given its path.
	 *  \param[in] name Name of this file. This will be used as the underlying destination filename.
	 *  \param[in,out] path Path to the file. When reading the archive, this is set to the extracted file path.
	 */
	void file(QString const& name, QString& path);

	/*! \brief Declare an error in the archive (while reading it) that can be repaired by further user actions.
	 */
	void repairable_error(boost::shared_ptr<PVSerializeArchiveFixError> const& error);

protected:
	void error_fixed(PVSerializeArchiveFixError* error);
	void fix_attribute(QString const& name, QVariant const& v);
	inline const void* bound_obj() const { return _bound_obj; }

private:
	p_type create_object(QString const& name, QString const& desc = QString(), bool optional = false, bool visible = true, bool def_option = true);
	uint32_t get_version() const;
	void attribute_write(QString const& name, QVariant const& obj);
	void attribute_read(QString const& name, QVariant& obj, QVariant const& def);
	void list_attributes_write(QString const& name, std::vector<QVariant> const& list);
	void list_attributes_read(QString const& name, std::vector<QVariant>& list);
	void hash_arguments_write(QString const& name, PVArgumentList const& obj);
	void hash_arguments_read(QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);
	bool must_write_child(QString const& name);
	p_type get_archive_object_from_path(QString const& path) const;

	template <typename T>
	void call_serialize(T& obj, p_type new_obj, T const* /*def_v*/) { obj.serialize(*new_obj, get_version()); new_obj->_bound_obj = &obj; new_obj->_bound_obj_type = typeid(T); }

	template <typename T>
	void call_serialize(T* obj, p_type new_obj, T const* def_v) { call_serialize(*obj, new_obj, def_v); }

	template <typename T>
	void call_serialize(boost::shared_ptr<T>& obj, p_type new_obj, T const* def_v)
	{
		if (!obj) {
			assert(!is_writing());
			T* new_p;
			if (def_v) {
				new_p = new T(*def_v);
			}
			else {
				new_p = new T();
			}
			obj.reset(new_p);
		}
		obj->serialize(*new_obj, get_version());
		new_obj->_bound_obj = obj.get();
		new_obj->_bound_obj_type = typeid(T);
	}

	template <typename T>
	void call_serialize(PVSharedPtr<T>& obj, p_type new_obj, T const*)
	{
		if (!obj) {
			assert(!is_writing());
			T* new_p;
			new_p = new T();
			obj.reset(new_p);
		}
		obj->serialize(*new_obj, get_version());
		new_obj->_bound_obj = obj.get();
		new_obj->_bound_obj_type = typeid(T);
	}

	template <typename T>
	void call_serialize(PVDataTreeAutoShared<T>& obj, p_type new_obj, T const*)
	{
		obj->serialize(*new_obj, get_version());
		new_obj->_bound_obj = obj.get();
		new_obj->_bound_obj_type = typeid(T);
	}

	template <typename T>
	static void* obj_pointer(T& obj) { return &obj; }

	template <typename T>
	static void* obj_pointer(boost::shared_ptr<T>& obj) { return obj.get(); }

	template <typename T>
	void pointer_to_obj(void* p, T& obj)
	{
		T* dp = (T*) (p);
		obj = *dp;
	}

	template <typename T>
	void pointer_to_obj(void* p, boost::shared_ptr<T>& obj)
	{
		T* dp = (T*) (p);
		obj = dp->shared_from_this();
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

	/*! \brief If relevant, represents a pointer to the object that has been serialized
	 */
	void* _bound_obj;
	PVTypeInfo _bound_obj_type;

	/*! \brief Whether or not this object is exposed to the user (for options)
	 */
	bool _visible;
};

typedef PVSerializeObject::p_type PVSerializeObject_p;

template <typename T>
bool PVSerializeObject::object(QString const& name, T& obj, QString const& desc, bool optional, typename PVTypeTraits::remove_shared_ptr<T>::type const* def_v, bool visible, bool def_option, p_type* used_so)
{
	if (optional && is_writing()) {
		if (!must_write_child(name)) {
			return false;
		}
	}
	p_type new_obj;
	try {
		new_obj = create_object(name, desc, optional, visible, def_option);
	}
	catch (PVSerializeArchiveErrorNoObject &e) {
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

template <typename T>
PVSerializeObject::p_type PVSerializeObject::list(QString const& name, T& obj, QString const& desc, typename PVTypeTraits::remove_shared_ptr<typename T::value_type>::type const* def_v, QStringList const& descriptions, bool visible, bool elts_optional)
{
	return list<T, typename T::value_type>(name, obj, desc, def_v, descriptions, visible, elts_optional);
}

template <typename T, typename V>
PVSerializeObject::p_type PVSerializeObject::list(QString const& name, T& obj, QString const& desc, typename PVTypeTraits::remove_shared_ptr<V>::type const* def_v, QStringList const& descriptions, bool visible, bool elts_optional)
{
	if (elts_optional && is_writing() && !must_write_child(name)) {
		return p_type();
	}

	typedef typename PVCore::PVTypeTraits::pointer<V>::type Vp;
	typedef typename PVCore::PVTypeTraits::pointer<typename T::value_type>::type Lvp;
	PVSerializeObject_p list_obj;
	try {
		list_obj = create_object(name, desc, elts_optional, visible);
	}
	catch (PVSerializeArchiveErrorNoObject& e) {
		if (!elts_optional && !is_writing()) {
			throw e;
		}
		return p_type();
	}
#ifndef NDEBUG
	if (descriptions.size() > 0) {
		assert(descriptions.size() == (int) obj.size());
	}
#endif
	QString desc_;
	if (is_writing()) {
		typename T::iterator it;
		int idx = 0;
		for (it = obj.begin(); it != obj.end(); it++) {
			QString child_name = QString::number(idx);
			if (!(elts_optional && !list_obj->must_write_child(child_name))) {
				Vp v = PVCore::PVTypeTraits::dynamic_pointer_cast<Vp, Lvp>::cast(PVCore::PVTypeTraits::pointer<typename T::value_type&>::get((*it)));
				assert(v);
				if (descriptions.size() > 0) {
					desc_ = descriptions.at(idx);
				}
				PVSerializeObject_p new_obj = list_obj->create_object(child_name, desc_, elts_optional);
				call_serialize(v, new_obj, def_v);
			}
			idx++;
		}
	}
	else {
		obj.clear();
		int idx = 0;
		try {
			while (true) {
				V v;
				PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
				call_serialize(v, new_obj, def_v);
				obj.push_back(v);
				idx++;
			}
		}
		catch (PVSerializeArchiveErrorNoObject const& /*e*/) {
			return list_obj;
		}
	}
	return list_obj;
}

template <typename F>
PVSerializeObject::p_type PVSerializeObject::list_read(F const& func, QString const& name, QString const& desc, bool visible, bool elts_optional)
{
	assert(!is_writing());

	typedef decltype(func()) V;
	typedef typename PVTypeTraits::remove_shared_ptr<V>::type const def_t;
	PVSerializeObject_p list_obj;
	try {
		list_obj = create_object(name, desc, elts_optional, visible);
	}
	catch (PVSerializeArchiveErrorNoObject& e) {
		if (!elts_optional && !is_writing()) {
			throw e;
		}
		return p_type();
	}
	int idx = 0;
	try {
		while (true) {
			PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			// This is really important to have the following line (creation of a new object) after
			// the creation of the PVSerializeObject_p. Indeed, when the exception of a "not found element" is thrown,
			// we would have potentially created a useless object (that might have been added, for instance, to the datatree).
			// If you want to store the children to a list, add the child to the list in the func function and return its address
			V v(func());
			call_serialize(v, new_obj, (def_t*) NULL);
			idx++;
		}
	}
	catch (PVSerializeArchiveErrorNoObject const&) {
		return list_obj;
	}
	return list_obj;
}

template <typename T>
void PVSerializeObject::list_ref(QString const& name, T& obj, p_type ref_so)
{
	if (is_writing()) {
		typename T::iterator it;
		QStringList ref_paths;
		for (it = obj.begin(); it != obj.end(); it++) {
			typename T::value_type& v = *it;

			// Look for `v' in `ref_so' children
			list_childs_t const& ref_children = ref_so->childs();
			list_childs_t::const_iterator it_child;
			PVSerializeObject_p found_ref;
			for (it_child = ref_children.begin(); it_child != ref_children.end(); it_child++) {
				PVSerializeObject_p test_so = it_child.value();
				assert(test_so->_bound_obj);
				if (obj_pointer(v) == test_so->_bound_obj) {
					found_ref = *it_child;
					break;
				}
			}
			// In this version, every elements of T must have already been serialized.
			assert(found_ref);
			ref_paths << found_ref->get_logical_path();
		}
		// Save the logical path references as a list of attributes
		list_attributes(name, ref_paths);
	}
	else {
		// Get the list of reference paths
		QStringList ref_paths;
		list_attributes(name, ref_paths);
		obj.clear();

		// Get the objects that must have been read from a previous serialization
		for (int i = 0; i < ref_paths.size(); i++) {
			PVSerializeObject_p obj_ref_so = get_archive_object_from_path(ref_paths[i]);
			assert(obj_ref_so->_bound_obj);
			typename T::value_type v;
			pointer_to_obj(obj_ref_so->_bound_obj, v);
			obj.push_back(v);
		}
	}
}

template <typename K, typename V>
void PVSerializeObject::hash(QString const& name, QHash<K,V>& obj)
{
	if (is_writing()) {
		typename QHash<K,V>::const_iterator it;
		for (it = obj.begin(); it != obj.end(); it++) {
			PVSerializeObject_p new_obj = create_object(it.key());
			it.value().serialize(*new_obj);
		}
	}
	else {
	}
}

template <class T>
void PVSerializeObject::attribute(QString const& name, T& obj, T const& def)
{
	if (is_writing()) {
		attribute_write(name, QVariant(obj));
	}
	else {
		QVariant v;
		attribute_read(name, v, QVariant(def));
		obj = v.value<T>();
	}
}

template <class T>
void PVSerializeObject::list_attributes(QString const& name, T& obj)
{
	list_attributes(name, obj, [=](QVariant const& v) { return v.value<typename T::value_type>(); });
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
	}
	else {
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
#define PVSERIALIZEOBJECT_SPLIT\
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)\
	{\
		so.split(*this);\
	}\

#endif
