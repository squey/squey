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

#include <boost/shared_ptr.hpp>

#include <pvkernel/core/general.h>

namespace PVCore
{
/*! \brief Special class to represent the fact that a tree object is at the root of the hierarchy.
*/
template <typename Tchild>
struct PVDataTreeNoParent
{
	typedef PVDataTreeNoParent parent_t;
	typedef Tchild child_t;
	typedef boost::shared_ptr<child_t> pchild_t;
	typedef QList<pchild_t> children_t;
	inline PVDataTreeNoParent* get_parent()
	{
		std::cout << "WARNING: The data tree object has no ancestor of such specified type" << std::endl;
		assert(false);
		return nullptr;
	}
	bool is_root() { return true; }
	void add_child(Tchild* /*child*/) {}
	pchild_t remove_child(Tchild* /*child*/) { return pchild_t();}
	children_t _children;
};

/*! \brief Special class to represent the fact that a tree object is not meant to have any children.
*/
template <typename Tparent>
struct PVDataTreeNoChildren
{
	typedef Tparent parent_t;
	typedef PVDataTreeNoChildren child_t;
	typedef boost::shared_ptr<child_t> pchild_t;
	typedef QList<pchild_t> children_t;
	inline PVDataTreeNoChildren* get_children()
	{
		std::cout << "WARNING: The data tree object has no children of such specified type" << std::endl;
		assert(false);
		return nullptr;
	}
	void dump(uint32_t /*spacing*/){}
	children_t _children;
};

/*! \brief Data tree object base class.
 *
 * This class is the base class for all objects of the data tree.
 */
template <typename Tparent, typename Tchild>
class PVDataTreeObject
{
public:
	template<typename T1, typename T2> friend class PVDataTreeObject;

public:
	typedef Tparent parent_t;
	typedef Tchild child_t;
	typedef boost::shared_ptr<child_t> pchild_t;
	typedef QList<pchild_t> children_t;

public:
	/*! \brief Create a TreeObject with the specified parent and add itself as one of its children.
	 *  \param[in] parent Parent of the object
	 *  The lifetime of the child is handled by the parent.
	 */
	PVDataTreeObject(Tparent* parent = nullptr) : _parent(parent)
	{
		// It's safe to use 'this' in the constructor since we just want
		// to store the address of the child in the parent.
		if (_parent) {
			auto me = static_cast<typename Tchild::parent_t*>(this);
			_parent->add_child(me);
		}
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	~PVDataTreeObject()
	{
		auto me = static_cast<typename Tchild::parent_t*>(this);
		std::cout << typeid(typename Tchild::parent_t).name() << "(" << me << ")"<< "::~TreeObject" << std::endl;
	}

	/*! \brief Check if the object is an instance of the root class of the hierarchy.
	 *  \return true, false otherwise.
	 */
	bool is_root()
	{
		return  std::is_same<Tparent, PVDataTreeNoParent<typename Tchild::parent_t> >::value;
	}

	/*! \brief Return an ancestor of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the parent is returned.
	 *  \return An ancestor.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename Tancestor = Tparent>
	inline Tancestor* get_parent()
	{
		return GetParentImpl<Tparent, Tancestor>::get_parent(_parent);
	}
	template <typename Tancestor = Tparent>
	inline const Tancestor* get_parent()  const
	{
		return GetParentImpl<Tparent, Tancestor>::get_parent(_parent);
	}

	/*! \brief Set the parent of a data tree object.
	 *  \param[in] parent Parent of the data tree object.
	 *  If a parent is already set, properly reparent with taking care of the child.
	 */
	void set_parent(Tparent* parent)
	{
		if (_parent == parent) {
			return;
		}

		auto me = static_cast<typename Tchild::parent_t*>(this);
		typename Tparent::pchild_t me_p;
		if (_parent) {
			me_p = _parent->remove_child(me);
			if (parent) {
				parent->_children.push_back(me_p);
			}
		}
		auto old_parent = _parent;
		_parent = parent;
		if (old_parent == nullptr && !parent->is_root()) {
			me_p.reset(me);
			parent->_children.push_back(me_p);
		}
	}

	/*! \brief Return the children of a data tree object at the specified hierarchical level (as a class type).
	 *  If no level is specified, the direct children are returned.
	 *  \return The list of children.
	 *  Note: Compile with '-std=c++0x' flag to support function template default parameter.
	 */
	template <typename T = Tchild>
	typename T::parent_t::children_t get_children()
	{
		return GetChildrenImpl<Tchild, T>::get_children(_children);
	}
	template <typename T = Tchild>
	const typename T::parent_t::children_t get_children() const
	{
		return GetChildrenImpl<Tchild, T>::get_children(_children);
	}

	/*! \brief Add a child to the data tree object.
	 *  \param[in] child Child of the data tree object to add.
	 *  A check is done to assert that the child is not already a children of the data tree object
	 *  in order to avoid a nasty mess.
	 *  If the child belongs to another hierarchy, stole it from its parent first.
	 */
	void add_child(Tchild* child)
	{
		if(child == nullptr) {
			return;
		}

		bool found = false;
		for (auto c: _children) {
			if (child == c.get())
			{
				found = true;
				break;
			}
		}
		if (!found) {
			auto me = static_cast<typename Tchild::parent_t*>(this);
			auto child_parent = child->get_parent();
			pchild_t pchild;
			if (child_parent && child_parent != me) {
				// steal child to parent
				pchild = child_parent->remove_child(child);
			}
			else {
				pchild.reset(child);
			}
			child->_parent = me;
			_children.push_back(pchild);
		}
		else {
			PVLOG_WARN("Tried to add child (0x%x) twice!\n", child);
			assert(false);
		}
	}


	void add_child(pchild_t child_p)
	{
		bool found = false;
		for (auto c: _children) {
			if (child_p.get() == c.get())
			{
				found = true;
				break;
			}
		}
		if (!found) {
			auto me = static_cast<typename Tchild::parent_t*>(this);
			auto child_parent = child_p->get_parent();
			pchild_t pchild;
			if (child_parent && child_parent != me) {
				// steal child to parent
				child_p = child_parent->remove_child(child_p.get());
			}
			child_p.get()->_parent = me;
			_children.push_back(child_p);
		}
		else {
			PVLOG_WARN("Tried to add child (0x%x) twice!\n", child_p.get());
			assert(false);
		}
	}

	/*! \brief Remove a child of the data tree object.
	 *  \param[in] child Child of the data tree object to remove.
	 *  \return a shared_ptr to the removed child in order to postpone its destruction.
	 */
	pchild_t remove_child(Tchild* child)
	{
		pchild_t pchild;
		for (int i = 0; i < _children.size(); i++) {
			if (child == _children[i].get())
			{
				pchild = _children[i];
				_children.erase(_children.begin()+i);
				pchild.get()->_parent = nullptr;
				break;
			}
		}

		return pchild;
	}

	/*! \brief Dump the data tree object and all of it's underlying children hierarchy.
	 */
	void dump(uint32_t spacing = 20)
	{
		auto me = static_cast<typename Tchild::parent_t*>(this);
		std::cout << " |" << std::setfill('-') << std::setw(spacing) << typeid(typename Tchild::parent_t).name() << "(" << me << ")" << std::endl;
		for (auto child: _children) {
			child->dump(spacing + 10);
		}
	}

	template <typename T = Tchild>
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
	Tparent* _parent;
	children_t _children;
};
}

#endif /* PVDATATREEOBJECT_H_ */
