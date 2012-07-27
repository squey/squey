/**
 * \file PVDataTreeObject.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVDATATREEOBJECT_H_
#define PVDATATREEOBJECT_H_

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>

#include <QList>

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>
#include <pvkernel/core/PVTypeTraits.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeObject.h>

namespace PVCore
{

// Helper class
template <class T>
class PVDataTreeAutoShared: public PVCore::PVSharedPtr<T>
{
	typedef typename T::parent_t parent_t;
	typedef typename T::pparent_t pparent_t;
	typedef PVDataTreeAutoShared<parent_t> auto_pparent_t;
public:
	template <typename... Tparams>
	PVDataTreeAutoShared(Tparams... params):
		PVCore::PVSharedPtr<T>(new T(params...))
	{ }

	PVDataTreeAutoShared(auto_pparent_t const& parent):
		PVCore::PVSharedPtr<T>(new T())
	{
		this->get()->set_parent(parent);
	}

	PVDataTreeAutoShared(pparent_t const& parent):
		PVCore::PVSharedPtr<T>(new T())
	{
		this->get()->set_parent(parent);
	}

	template <typename... Tparams>
	PVDataTreeAutoShared(pparent_t const& parent, Tparams... params):
		PVCore::PVSharedPtr<T>(new T(params...))
	{
		this->get()->set_parent(parent);
	}

	template <typename... Tparams>
	PVDataTreeAutoShared(auto_pparent_t const& parent, Tparams... params):
		PVCore::PVSharedPtr<T>(new T(params...))
	{
		this->get()->set_parent(parent);
	}

public:
	PVDataTreeAutoShared(PVCore::PVSharedPtr<T> const& o):
		PVCore::PVSharedPtr<T>(o)
	{ }

public:
	static PVDataTreeAutoShared<T> invalid() { return PVCore::PVSharedPtr<T>(); }

public:
	// Implicit conversion to PVCore::PVSharedPtr<T>
	inline operator PVCore::PVSharedPtr<T>&() { return *this; }
	inline operator PVCore::PVSharedPtr<T> const&() const { return *this; }
	inline PVDataTreeAutoShared& operator=(PVDataTreeAutoShared<T> const& o) { PVCore::PVSharedPtr<T>::operator=(static_cast<PVCore::PVSharedPtr<T> const&>(o)); return *this; }
	inline PVDataTreeAutoShared& operator=(PVCore::PVSharedPtr<T> const& o) { PVCore::PVSharedPtr<T>::operator=(o); return *this; }
};

namespace PVTypeTraits {

template <class T>
struct remove_shared_ptr<PVDataTreeAutoShared<T> >
{
	typedef T type;
};

template <class T>
struct pointer<PVDataTreeAutoShared<T> >
{
	typedef PVDataTreeAutoShared<T> type;
	static inline type get(type obj) { return obj; }
};

template <class T>
struct pointer< PVDataTreeAutoShared<T>& >
{
	typedef  PVDataTreeAutoShared<T>& type;
	static inline type get(type obj) { return obj; }
};

template <class Y, class T>
struct dynamic_pointer_cast< PVDataTreeAutoShared<Y>,  PVDataTreeAutoShared<T> >
{
	typedef PVDataTreeAutoShared<T> org_pointer;
	typedef PVDataTreeAutoShared<Y> result_pointer;
	static result_pointer cast(org_pointer const& p) { return result_pointer(PVCore::dynamic_pointer_cast<Y>(p)); }
};

}

/*! \brief Special class to represent the fact that a tree object is at the root of the hierarchy.
*/
template <typename Treal>
struct PVDataTreeNoParent
{
	typedef PVDataTreeNoParent parent_t;
	typedef PVCore::PVSharedPtr<PVDataTreeNoParent> pparent_t;
	typedef Treal child_t;
	typedef PVCore::PVSharedPtr<child_t> pchild_t;
	typedef QList<pchild_t> children_t;

	inline PVDataTreeNoParent* get_parent()
	{
		std::cout << "WARNING: The data tree object has no ancestor of such specified type" << std::endl;
		assert(false);
		return nullptr;
	}
	pchild_t remove_child(child_t&) { return pchild_t(); }
	void do_add_child(pchild_t const&) { }
};

template <class T>
class PVDataTreeAutoShared<PVDataTreeNoParent<T> > : public PVCore::PVSharedPtr<PVDataTreeNoParent<T> >
{
	typedef PVDataTreeNoParent<T> obj_type;
	typedef PVCore::PVSharedPtr<obj_type> p_type;
public:
	template <typename... Tparams>
	PVDataTreeAutoShared(Tparams... params):
		p_type(new obj_type(params...))
	{ }

public:
	PVDataTreeAutoShared(p_type const& o):
		p_type(o)
	{ }

public:
	// Implicit conversion to PVCore::PVSharedPtr<T>
	operator p_type&() { return *this; }
	operator p_type const&() const { return *this; }

	template <class Y>
	operator PVCore::PVSharedPtr<Y>&() { return PVCore::static_pointer_cast<Y>(*this); }
	template <class Y>
	operator PVCore::PVSharedPtr<Y> const&() const { return PVCore::static_pointer_cast<Y>(*this); }

};

/*! \brief Special class to represent the fact that a tree object is not meant to have any children.
*/
template <typename Tparent>
struct PVDataTreeNoChildren
{
	typedef Tparent parent_t;
	typedef PVDataTreeNoChildren child_t;
	typedef PVDataTreeAutoShared<child_t> pchild_t;
	typedef QList<pchild_t> children_t;
	typedef PVCore::PVSharedPtr<parent_t> pparent_t;

	inline PVDataTreeNoChildren* get_children()
	{
		std::cout << "WARNING: The data tree object has no children of such specified type" << std::endl;
		assert(false);
		return nullptr;
	}
	void dump(uint32_t){}
	void serialize(PVCore::PVSerializeObject&, PVCore::PVSerializeArchive::version_t) {}
	void set_parent(parent_t*) {}
	void set_parent(pparent_t const&) {}
	QString get_serialize_description() { return QString(); }

	children_t _children;
};

/*! \brief Data tree object base class.
 *
 * This class is the base class for all objects of the data tree.
 */
template <typename Tparent, typename Tchild>
class PVDataTreeObject: public PVEnableSharedFromThis<typename Tparent::child_t >
{
public:
	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	typedef Tparent parent_t;
	typedef Tchild child_t;
	typedef PVCore::PVSharedPtr<parent_t> pparent_t;
	typedef PVDataTreeAutoShared<child_t> pchild_t;
	typedef QList<pchild_t> children_t;
	typedef PVDataTreeObject<parent_t, child_t> data_tree_t;

private:
	typedef typename parent_t::child_t real_type_t;

public:
	typedef PVDataTreeAutoShared<real_type_t> p_type;

public:
	/*! \brief Default constructor
	 */
	PVDataTreeObject():
		PVEnableSharedFromThis<real_type_t>(),
		_parent(nullptr)
	{
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	virtual ~PVDataTreeObject() {}

	/*! \brief Return an ancestor of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the parent is returned.
	 *  \return An ancestor.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename Tancestor = parent_t>
	inline Tancestor* get_parent()
	{
		return GetParentImpl<parent_t, Tancestor>::get_parent(_parent);
	}
	template <typename Tancestor = parent_t>
	inline const Tancestor* get_parent()  const
	{
		return GetParentImpl<parent_t, Tancestor>::get_parent(_parent);
	}

	inline void set_parent(pparent_t const& parent) { set_parent_from_ptr(parent.get()); }
	inline void set_parent(parent_t* parent) { set_parent_from_ptr(parent); }

	template <typename... Tparams>
	child_t* new_child(Tparams... params)
	{
		pchild_t ret(new child_t(params...));
		add_child(ret);
		return ret;
	}

	/*! \brief Return the children of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the direct children are returned.
	 *  \return The list of children.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename T = child_t>
	typename T::parent_t::children_t get_children()
	{
		return GetChildrenImpl<child_t, T>::get_children(_children);
	}
	template <typename T = child_t>
	const typename T::parent_t::children_t get_children() const
	{
		return GetChildrenImpl<child_t, T>::get_children(_children);
	}

	inline children_t const& get_children() const { return _children; }

	/*! \brief Add a child to the data tree object.
	 *  \param[in] child Child of the data tree object to add.
	 *  This is basically a helper method doing a set_parent on the child.
	 */
	void add_child(pchild_t const& child_p)
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		child_p->set_parent(me);
	}

	/*! \brief Remove a child of the data tree object.
	 *  \param[in] child Child of the data tree object to remove.
	 *  \return a shared_ptr to the removed child in order to postpone its destruction if wanted.
	 */
	pchild_t remove_child(child_t const& child)
	{
		PVCore::PVSharedPtr<child_t> pchild;
		for (int i = 0; i < _children.size(); i++) {
			if (&child == _children[i].get()) {
				//child_about_to_be_removed(child);
				pchild = _children[i];
				child_about_to_be_removed(*pchild);
				_children.erase(_children.begin()+i);
				pchild->_parent = nullptr;
				break;
			}
		}

		return pchild;
	}
	inline pchild_t remove_child(pchild_t const& child) { return remove_child(*child); }

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		QStringList descriptions;
		for (auto child : _children) {
			descriptions << child->get_serialize_description();
		}
		so.list(get_children_serialize_name(), _children, get_children_description(), (child_t*) NULL, descriptions);
	};

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
	{
		PVCore::PVSharedPtr<real_type_t> me_p(static_cast<real_type_t*>(this)->shared_from_this());
		auto create_func = [&]{ return PVDataTreeAutoShared<child_t>(me_p); };
		if (!so.list_read(create_func, get_children_serialize_name(), get_children_description(), true, true)) {
			// No children born in here...
			return;
		}
	}

	/*! \brief Dump the data tree object and all of it's underlying children hierarchy.
	 */
	void dump(uint32_t spacing = 20)
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		std::cout << " |" << std::setfill('-') << std::setw(spacing) << typeid(real_type_t).name() << "(" << me << ")" << std::endl;
		for (auto child: _children) {
			child->dump(spacing + 10);
		}
	}

	template <typename T = child_t>
	void dump_children()
	{
		auto children = get_children<T>();
		std::cout << "(";
		for (int i = 0; i < children.size() ; i++) {
			if(i != 0)
				std::cout << ", ";
			std::cout << children[i];
		}
		std::cout << ")" << std::endl;
	}

protected:
	/*! \brief Set the parent of a data tree object.
	 *  \param[in] parent Parent of the data tree object.
	 *  If a parent is already set, properly reparent with taking care of the child.
	 */
	virtual void set_parent_from_ptr(parent_t* parent)
	{
		if (_parent == parent) {
			return;
		}

		real_type_t* me = static_cast<real_type_t*>(this);
		PVCore::PVSharedPtr<real_type_t> me_p;
		bool child_added = false;
		if (_parent) {
			me_p = _parent->remove_child(*me);
			if (parent) {
				parent->do_add_child(me_p);
				child_added = true;
			}
		}
		parent_t* old_parent = _parent;
		_parent = parent;
		if (old_parent == nullptr && parent && !child_added) {
			me_p = PVCore::static_pointer_cast<real_type_t>(me->shared_from_this()); // ??
			parent->do_add_child(me_p);
		}
	}

	void do_add_child(pchild_t c)
	{
		_children.push_back(c);
		child_added(*c);
	}


	virtual QString get_serialize_description() const { return QString(); }
	virtual QString get_children_description() const { return "Children"; }
	virtual QString get_children_serialize_name() const { return "children"; }

protected:
	// Events
	virtual void child_added(child_t& /*child*/) { }
	virtual void child_about_to_be_removed(child_t& /*child*/) { }

private:
	/*! \brief Implementation of the PVDataTreeObject::get_parent() method.
	 *	"function template partial specialization is not allowed":
	 * 	we must use static methods inside a template class
	 */
	template <typename T, typename Tancestor, bool B = std::is_same<T, Tancestor>::value>
	struct GetParentImpl;

	/*! \brief Partial specialization for the case we have found the ancestor.
	 */
	template <typename T, typename Tancestor>
	struct GetParentImpl<T, Tancestor, true>
	{
		static inline Tancestor* get_parent(Tancestor* parent)
		{
			return parent;
		}
	};

	/*! \brief Partial specialization for the case we haven't found the ancestor yet.
	 * 	Recursively specialize this class with a parent of one level higher.
	 */
	template <typename T, typename Tancestor>
	struct GetParentImpl<T, Tancestor, false>
	{
		static inline Tancestor* get_parent(T* parent)
		{
			if (parent != nullptr) {
				return GetParentImpl<typename T::parent_t, Tancestor>::get_parent(parent->get_parent());
			}

			return nullptr;
		}
	};

	/*! \brief Implementation of the PVDataTreeObject::get_children() method.
	 *	"function template partial specialization is not allowed":
	 * 	we must use static methods inside a template class
	 */
	template <typename T, typename Tc, bool B = std::is_same<T, Tc>::value>
	struct GetChildrenImpl;

	/*! \brief Partial specialization for the case we have found the child class.
	*/
	template <typename T, typename Tc>
	struct GetChildrenImpl<T, Tc, true>
	{
		static inline typename Tc::parent_t::children_t get_children(typename Tc::parent_t::children_t children)
		{
			return children;
		}
	};

	/*! \brief Partial specialization for the case we haven't found the child class yet.
	 * 	Recursively specialize this class with a child class of one level lower.
	 */
	template <typename T, typename Tc>
	struct GetChildrenImpl<T, Tc, false>
	{
		static inline typename Tc::parent_t::children_t get_children(typename T::parent_t::children_t children)
		{
			typename T::children_t children_tmp;
			for (auto child : children) {
				for (auto c : child->get_children()) {
					children_tmp.push_back(c);
				}
			}
			return GetChildrenImpl<typename T::child_t, Tc>::get_children(std::move(children_tmp));
		}
	};

private:
	parent_t* _parent;
	children_t _children;
};

#if 0
// Specialcase when root item !
template <typename Troot, typename Tchild>
class PVDataTreeObject<PVDataTreeNoParent<Troot>, Tchild>: public PVSharedPtr::enable_shared_from_this<Troot>
{
public:
	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	typedef Tchild child_t;
	typedef PVDataTreeAutoShared<child_t> pchild_t;
	typedef QList<pchild_t> children_t;
	typedef PVDataTreeObject<PVDataTreeNoParent<Troot>, child_t> data_tree_t;
	static constexpr bool has_parent = false;

private:
	typedef Troot real_type_t;

public:
	typedef PVDataTreeAutoShared<real_type_t> p_type;

public:
	/*! \brief Default constructor
	 */
	PVDataTreeObject():
		PVSharedPtr::enable_shared_from_this<real_type_t>()
	{
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	virtual ~PVDataTreeObject() {}

	virtual QString get_serialize_description() { return QString(); }
	virtual QString get_children_description() { return "Children"; }

	void* get_parent() const { return nullptr; }

	template <typename... Tparams>
	child_t* new_child(Tparams... params)
	{
		pchild_t ret(new child_t(params...));
		add_child(ret);
		return ret;
	}

	/*! \brief Return the children of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the direct children are returned.
	 *  \return The list of children.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename T = child_t>
	typename T::parent_t::children_t get_children()
	{
		return GetChildrenImpl<child_t, T>::get_children(_children);
	}
	template <typename T = child_t>
	const typename T::parent_t::children_t get_children() const
	{
		return GetChildrenImpl<child_t, T>::get_children(_children);
	}

	inline children_t const& get_children() { return _children; }

	/*! \brief Add a child to the data tree object.
	 *  \param[in] child Child of the data tree object to add.
	 *  This is basically a helper method doing a set_parent on the child.
	 */
	void add_child(pchild_t& child_p)
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		child_p->set_parent(me);
	}

	/*! \brief Remove a child of the data tree object.
	 *  \param[in] child Child of the data tree object to remove.
	 *  \return a shared_ptr to the removed child in order to postpone its destruction if wanted.
	 */
	pchild_t remove_child(child_t const& child)
	{
		pchild_t pchild;
		for (int i = 0; i < _children.size(); i++) {
			if (&child == _children[i].get()) {
				//child_about_to_be_removed(child);
				pchild = _children[i];
				child_about_to_be_removed(*pchild);
				_children.erase(_children.begin()+i);
				pchild->_parent = nullptr;
				break;
			}
		}

		return pchild;
	}
	inline pchild_t remove_child(pchild_t const& child) { return remove_child(*child); }

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		QStringList descriptions;
		for (auto child : _children) {
			descriptions << child->get_serialize_description();
		}
		so.list(get_children_description(), _children, QString(), (child_t*) NULL, descriptions);
	};

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
	{
		auto create_func = [&]{ return PVDataTreeAutoShared<child_t>(this->shared_from_this()); };
		if (!so.list_read(create_func, get_children_description(), QString(), true, true)) {
			// No children born in here...
			return;
		}
	}

	/*! \brief Dump the data tree object and all of it's underlying children hierarchy.
	 */
	void dump(uint32_t spacing = 20)
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		std::cout << " |" << std::setfill('-') << std::setw(spacing) << typeid(real_type_t).name() << "(" << me << ")" << std::endl;
		for (auto child: _children) {
			child->dump(spacing + 10);
		}
	}

	template <typename T = child_t>
	void dump_children()
	{
		auto children = get_children<T>();
		std::cout << "(";
		for (int i = 0; i < children.size() ; i++) {
			if(i != 0)
				std::cout << ", ";
			std::cout << children[i];
		}
		std::cout << ")" << std::endl;
	}

	void do_add_child(pchild_t c)
	{
		_children.push_back(c);
		child_added(*c);
	}

protected:
	// Events
	virtual void child_added(child_t& /*child*/) { }
	virtual void child_about_to_be_removed(child_t& /*child*/) { }

private:
	/*! \brief Implementation of the PVDataTreeObject::get_children() method.
	 *	"function template partial specialization is not allowed":
	 * 	we must use static methods inside a template class
	 */
	template <typename T, typename Tc, bool B = std::is_same<T, Tc>::value>
	struct GetChildrenImpl;

	/*! \brief Partial specialization for the case we have found the child class.
	*/
	template <typename T, typename Tc>
	struct GetChildrenImpl<T, Tc, true>
	{
		static inline typename Tc::parent_t::children_t get_children(typename Tc::parent_t::children_t children)
		{
			return children;
		}
	};

	/*! \brief Partial specialization for the case we haven't found the child class yet.
	 * 	Recursively specialize this class with a child class of one level lower.
	 */
	template <typename T, typename Tc>
	struct GetChildrenImpl<T, Tc, false>
	{
		static inline typename Tc::parent_t::children_t get_children(typename T::parent_t::children_t children)
		{
			typename T::children_t children_tmp;
			for (auto child : children) {
				for (auto c : child->get_children()) {
					children_tmp.push_back(c);
				}
			}
			return GetChildrenImpl<typename T::child_t, Tc>::get_children(std::move(children_tmp));
		}
	};

private:
	children_t _children;
};
#endif

}

#endif /* PVDATATREEOBJECT_H_ */
