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

#include <pvkernel/core/PVDataTreeAutoShared.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>
#include <pvkernel/core/PVTypeTraits.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeObject.h>

namespace PVCore
{

/*! \brief Special class to represent the fact that a tree object is at the root of the hierarchy.
*/
template <typename Treal>
struct PVDataTreeNoParent
{
};

/*! \brief Special class to represent the fact that a tree object is not meant to have any children.
*/
template <typename Treal>
struct PVDataTreeNoChildren
{
};

class PVDataTreeObjectBase;

class PVDataTreeObjectWithParentBase
{
public:
	PVDataTreeObjectWithParentBase():
		_parent(nullptr)
	{ }

public:
	inline PVDataTreeObjectBase* get_parent_base() { return _parent; }
	inline const PVDataTreeObjectBase* get_parent_base() const { return _parent; }

protected:
	PVDataTreeObjectBase* _parent;
};

class PVDataTreeObjectWithChildrenBase
{
public:
	typedef QList<PVDataTreeObjectBase*> children_base_t;
public:
	// This is virtual and shouldn't be. Children should be stored as a list of shared_ptr to
	// PVDataTreeObjectBase and converted when necessary.
	virtual children_base_t get_children_base() const = 0;
	virtual size_t get_children_count() const = 0;
};

class PVDataTreeObjectBase
{
public:
	typedef PVCore::PVSharedPtr<PVDataTreeObjectBase> base_p_type;
	typedef PVCore::PVSharedPtr<PVDataTreeObjectBase const> const_base_p_type;
public:
	virtual ~PVDataTreeObjectBase() { }

public:
	PVDataTreeObjectWithParentBase const* cast_with_parent() const { return dynamic_cast<PVDataTreeObjectWithParentBase const*>(this); }
	PVDataTreeObjectWithParentBase*       cast_with_parent()       { return dynamic_cast<PVDataTreeObjectWithParentBase*>(this); }

	PVDataTreeObjectWithChildrenBase const* cast_with_children() const { return dynamic_cast<PVDataTreeObjectWithChildrenBase const*>(this); }
	PVDataTreeObjectWithChildrenBase*       cast_with_children()       { return dynamic_cast<PVDataTreeObjectWithChildrenBase*>(this); }

public:
	virtual base_p_type base_shared_from_this() = 0;
	virtual const_base_p_type base_shared_from_this() const = 0;

public:
	template <class F>
	void depth_first_list(F const& f)
	{
		PVDataTreeObjectWithChildrenBase* obj_children = cast_with_children();
		if (!obj_children) {
			return;
		}
		for (PVDataTreeObjectBase* c : obj_children->get_children_base()) {
			f(c);
			c->depth_first_list(f);
		}
	}

public:
	virtual QString get_serialize_description() const { return QString(); }
};

namespace __impl {

template <typename Tparent, typename real_type_t>
class PVDataTreeObjectWithParent;

template <typename Tchild, typename real_type_t>
class PVDataTreeObjectWithChildren: public PVDataTreeObjectWithChildrenBase
{
	template <typename T1, typename T2> friend class PVDataTreeObjectWithParent;

public:
	typedef Tchild child_t;
	typedef PVDataTreeAutoShared<child_t> pchild_t;
	typedef QList<pchild_t> children_t;

public:
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
				//pchild->_parent = nullptr;
				break;
			}
		}

		return pchild;
	}
	inline pchild_t remove_child(pchild_t const& child) { return remove_child(*child); }

	template <typename T = child_t>
	void dump_children()
	{
		auto children = get_children<T>();
		std::cout << "(";
		for (int i = 0; i < children.size() ; i++) {
			if(i != 0)
				std::cout << ", ";
			std::cout << children[i] << " (" << children[i].use_count() << ")" << std::endl;
		}
		std::cout << ")" << std::endl;
	}

public:
	virtual children_base_t get_children_base() const
	{
		children_base_t ret;
		ret.reserve(_children.size());
		for (pchild_t const& c : _children) {
			ret.push_back(static_cast<PVDataTreeObjectBase*>(c.get()));
		}
		return ret;
	}

	virtual size_t get_children_count() const { return _children.size(); }

public:
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

public:
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

	virtual void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
	{
		if (so.is_writing()) {
			serialize_write(so);
		}
		else {
			serialize_read(so, v);
		}
	}


protected:
	void do_add_child(pchild_t c)
	{
		_children.push_back(c);
		child_added(*c);
	}


	virtual QString get_children_description() const { return "Children"; }
	virtual QString get_children_serialize_name() const { return "children"; }

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

template <typename Tparent, typename real_type_t>
class PVDataTreeObjectWithParent: public PVDataTreeObjectWithParentBase
{
	template <typename T1, typename T2> friend class PVDataTreeObjectWithChildren;

public:
	typedef Tparent parent_t;
	typedef PVCore::PVSharedPtr<parent_t> pparent_t;

public:
	PVDataTreeObjectWithParent():
		PVDataTreeObjectWithParentBase()
	{ }

public:
	/*! \brief Return an ancestor of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the parent is returned.
	 *  \return An ancestor.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename Tancestor = parent_t>
	inline Tancestor* get_parent()
	{
		static_assert(std::is_same<Tancestor, real_type_t>::value == false, "PVDataTreeObject::get_parent: one object is asking itself as a parent.");
		return GetParentImpl<parent_t, Tancestor>::get_parent(get_real_parent());
	}
	template <typename Tancestor = parent_t>
	inline const Tancestor* get_parent()  const
	{
		static_assert(std::is_same<Tancestor, real_type_t>::value == false, "PVDataTreeObject::get_parent: one object is asking itself as a parent.");
		return GetParentImpl<parent_t const, Tancestor const>::get_parent(get_real_parent());
	}

	inline void set_parent(pparent_t const& parent) { set_parent_from_ptr(parent.get()); }
	inline void set_parent(parent_t* parent) { set_parent_from_ptr(parent); }

	void remove_from_tree()
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		get_real_parent()->remove_child(*me);
	}

protected:
	/*! \brief Set the parent of a data tree object.
	 *  \param[in] parent Parent of the data tree object.
	 *  If a parent is already set, properly reparent with taking care of the child.
	 */
	virtual void set_parent_from_ptr(parent_t* parent)
	{
		if (get_real_parent() == parent) {
			return;
		}

		real_type_t* me = static_cast<real_type_t*>(this);
		PVCore::PVSharedPtr<real_type_t> me_p;
		bool child_added = false;
		if (get_real_parent()) {
			me_p = get_real_parent()->remove_child(*me);
			if (parent) {
				parent->do_add_child(me_p);
				child_added = true;
			}
		}
		parent_t* old_parent = get_real_parent();
		_parent = parent;
		if (old_parent == nullptr && parent && !child_added) {
			me_p = PVCore::static_pointer_cast<real_type_t>(me->shared_from_this());
			parent->do_add_child(me_p);
		}
	}

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
				return GetParentImpl<typename PVCore::PVTypeTraits::const_fwd<typename T::parent_t, T>::type, Tancestor>::get_parent(parent->get_parent());
			}

			return nullptr;
		}
	};

	/*! \brief Get parent as a parent_t object
	 */
	parent_t* get_real_parent() { return static_cast<parent_t*>(get_parent_base()); }
	parent_t const* get_real_parent() const { return static_cast<parent_t const*>(get_parent_base()); }
};

}

/*! \brief Data tree object base class.
 *
 * This class is the base class for all objects of the data tree.
 */
template <typename Tparent, typename Tchild>
class PVDataTreeObject: public PVEnableSharedFromThis<typename Tparent::child_t >,
                        public __impl::PVDataTreeObjectWithChildren<Tchild, typename Tparent::child_t>,
						public __impl::PVDataTreeObjectWithParent<Tparent, typename Tparent::child_t>,
						public PVDataTreeObjectBase
{
	typedef __impl::PVDataTreeObjectWithChildren<Tchild, typename Tparent::child_t> impl_children_t;
	typedef __impl::PVDataTreeObjectWithParent<Tparent, typename Tparent::child_t>  impl_parent_t;

	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	static constexpr bool has_parent   = true;
	static constexpr bool has_children = true;

public:
	typedef typename impl_children_t::child_t    child_t;
	typedef typename impl_children_t::pchild_t   pchild_t;
	typedef typename impl_children_t::children_t children_t;

	typedef typename impl_parent_t::parent_t  parent_t;
	typedef typename impl_parent_t::pparent_t pparent_t;

	typedef typename parent_t::root_t root_t;

	typedef PVDataTreeObject<parent_t, child_t> data_tree_t;

private:
	typedef typename parent_t::child_t real_type_t;

public:
	typedef PVDataTreeAutoShared<real_type_t> p_type;
	typedef PVCore::PVWeakPtr<real_type_t>   wp_type;

public:
	/*! \brief Default constructor
	 */
	PVDataTreeObject():
		PVEnableSharedFromThis<real_type_t>(),
		impl_children_t(),
		impl_parent_t()
	{
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	virtual ~PVDataTreeObject() {}

public:
	virtual base_p_type base_shared_from_this()
	{
		PVCore::PVSharedPtr<real_type_t> p(static_cast<real_type_t*>(this)->shared_from_this());
		return std::move(base_p_type(p));
	}
	virtual const_base_p_type base_shared_from_this() const
	{
		PVCore::PVSharedPtr<real_type_t const> p(static_cast<real_type_t const*>(this)->shared_from_this());
		return std::move(const_base_p_type(p));
	}
};

// Special case when root item !
template <typename Troot, typename Tchild>
class PVDataTreeObject<PVDataTreeNoParent<Troot>, Tchild>: public PVEnableSharedFromThis<Troot>,
                                                           public __impl::PVDataTreeObjectWithChildren<Tchild, Troot>,
														   public PVDataTreeObjectBase
{
	typedef __impl::PVDataTreeObjectWithChildren<Tchild, Troot> impl_children_t;
	typedef __impl::PVDataTreeObjectWithChildren<Tchild, Troot> impl_base_t;
	
	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	static constexpr bool has_parent   = false;
	static constexpr bool has_children = true;

public:
	typedef typename impl_children_t::child_t    child_t;
	typedef typename impl_children_t::pchild_t   pchild_t;
	typedef typename impl_children_t::children_t children_t;

	typedef PVDataTreeObject<PVDataTreeNoParent<Troot>, child_t> data_tree_t;

private:
	typedef Troot real_type_t;

public:
	typedef PVDataTreeAutoShared<real_type_t> p_type;
	typedef PVCore::PVWeakPtr<real_type_t>   wp_type;
	typedef real_type_t root_t;

public:
	virtual base_p_type base_shared_from_this()
	{
		PVCore::PVSharedPtr<real_type_t> p(static_cast<real_type_t*>(this)->shared_from_this());
		return std::move(base_p_type(p));
	}
	virtual const_base_p_type base_shared_from_this() const
	{
		PVCore::PVSharedPtr<real_type_t const> p(static_cast<real_type_t const*>(this)->shared_from_this());
		return std::move(const_base_p_type(p));
	}

public:
	/*! \brief Default constructor
	 */
	PVDataTreeObject():
		PVEnableSharedFromThis<real_type_t>(),
		impl_children_t()
	{
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	virtual ~PVDataTreeObject() {}
};

// Special case when no children !
template <typename Tparent, typename Treal>
class PVDataTreeObject<Tparent, PVDataTreeNoChildren<Treal> >: public PVEnableSharedFromThis<Treal>,
                                                               public __impl::PVDataTreeObjectWithParent<Tparent, Treal>,
															   public PVDataTreeObjectBase
{
	typedef __impl::PVDataTreeObjectWithParent<Tparent, Treal> impl_parent_t;

	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	static constexpr bool has_parent   = true;
	static constexpr bool has_children = false;

public:
	typedef PVDataTreeObject<Tparent, PVDataTreeNoChildren<Treal> > data_tree_t;

private:
	typedef Treal real_type_t;

public:
	typedef PVDataTreeAutoShared<real_type_t> p_type;
	typedef PVCore::PVWeakPtr<real_type_t>   wp_type;
	typedef Tparent parent_t;
	typedef typename parent_t::root_t root_t;

public:
	/*! \brief Default constructor
	 */
	PVDataTreeObject():
		PVEnableSharedFromThis<real_type_t>(),
		impl_parent_t()
	{
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	virtual ~PVDataTreeObject() {}

public:
	virtual base_p_type base_shared_from_this()
	{
		PVCore::PVSharedPtr<real_type_t> p(static_cast<real_type_t*>(this)->shared_from_this());
		return std::move(base_p_type(p));
	}
	virtual const_base_p_type base_shared_from_this() const
	{
		PVCore::PVSharedPtr<real_type_t const> p(static_cast<real_type_t const*>(this)->shared_from_this());
		return std::move(const_base_p_type(p));
	}

public:
	virtual void serialize(PVCore::PVSerializeObject& /*so*/, PVCore::PVSerializeArchive::version_t /*v*/) { }

public:
	/*! \brief Dump the data tree object and all of it's underlying children hierarchy.
	 */
	void dump(uint32_t spacing = 20)
	{
		real_type_t* me = static_cast<real_type_t*>(this);
		PVDataTreeObjectBase* base = static_cast<PVDataTreeObjectBase*>(this);
		std::cout << " |" << std::setfill('-') << std::setw(spacing) << typeid(real_type_t).name() << "(" << me << ", base: " << base << ")" << std::endl;
	}
};

}

#endif /* PVDATATREEOBJECT_H_ */
