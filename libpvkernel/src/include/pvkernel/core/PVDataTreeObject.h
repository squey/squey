#ifndef PVDATATREEOBJECT_H_
#define PVDATATREEOBJECT_H_

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include <vector>
using namespace std;

namespace PVCore
{
/*! \brief Data tree object base class.
 *
 * This class is the base class for all objects of the data tree.
 */
template <typename Tparent, typename Tchild>
class PVDataTreeObject
{
public:
	typedef Tparent parent_t;
	typedef std::vector<Tchild*> children_t;

public:
	/*! \brief Create a TreeObject with the specified parent and add itself as one of its children.
	 *  \param[in] parent Parent of the object
	 *  The lifetime of the child is handled by the parent.
	 */
	PVDataTreeObject(Tparent* parent = NULL) : _parent(parent)
	{
		// It's safe to use 'this' in the constructor since we just want
		// to store the address of the child in the parent.
		if(_parent){
			_parent->add_child(static_cast<typename Tchild::parent_t*>(this));
		}
	}

	/*! \brief Delete the data tree object and all of it's underlying children hierarchy.
	 */
	~PVDataTreeObject()
	{
		auto me = static_cast<typename Tchild::parent_t*>(this);
		_parent->remove_child(me);
		std::cout << typeid(typename Tchild::parent_t).name() << "(" << me << ")"<< "::~TreeObject" << std::endl;
		for (auto child: _children) {
			delete child;
		}
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

	/*! \brief Set the parent of a data tree object.
	 *  \param[in] parent Parent of the data tree object.
	 *  If a parent is already set, properly reparent with taking care of the child.
	 */
	void set_parent(Tparent* parent)
	{
		auto me = static_cast<typename Tchild::parent_t*>(this);
		if (_parent) {
			_parent->remove_child(me);
		}
		_parent = parent;
		_parent->add_child(me);
	}

	/*! \brief Add a child to the data tree object.
	 *  \param[in] child Child of the data tree object to add.
	 *  A check is done to assert that the child is not already a children of the data tree object
	 *  in order to avoid a nasty mess.
	 *  If the child belongs to another hierarchy, stole it from its parent first.
	 */
	void add_child(Tchild* child)
	{
		// TODO: use std::find instead
		bool already_exist = false;
		for (auto c: _children) {
			if (child == c)
			{
				already_exist = true;
			}
		}
		if (!already_exist) {
			auto me = static_cast<typename Tchild::parent_t*>(this);
			auto child_parent = child->get_parent();
			if (child_parent != me) {
				// steal child to parent
				child_parent->remove_child(child);
			}
			_children.push_back(child);
		}
		else {
			assert(false); // tried to add child twice!
		}
	}

	/*! \brief Helper method to add several children to the data tree object.
	 */
	void add_children(children_t children)
	{
		for (auto child: children) {
			add_child(child);
		}
	}

	/*! \brief Remove a child of the data tree object.
	 *  \param[in] child Child of the data tree object to remove.
	 */
	void remove_child(Tchild* child)
	{
		_children.erase(std::remove(_children.begin(), _children.end(), child), _children.end());
	}

	/*! \brief Dump the data tree object and all of it's underlying children hierarchy.
	 */
	void dump(uint32_t spacing = 10)
	{
		auto me = static_cast<typename Tchild::parent_t*>(this);
		std::cout << " |" << std::setfill('-') << std::setw(spacing) << typeid(typename Tchild::parent_t).name() << "(" << me << ")" << std::endl;
		for (auto child: _children) {
			child->dump(spacing + 5);
		}
	}

private:
	/*! \brief Implementation of the TreeObject::get_parent() method.
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
		static inline Tancestor* get_parent(Tancestor* parent) { return parent; }
	};

	/*! \brief Partial specialization for the case we haven't found the ancestor yet.
	 * 	Recursively specialize this class with a parent of one level higher.
	 */
	template <typename T, typename Tancestor>
	struct GetParentImpl<T, Tancestor, false>
	{
		typedef typename T::parent_t parent_t;
		static inline Tancestor* get_parent(T* parent)
		{
			return GetParentImpl<parent_t, Tancestor>::get_parent(parent->get_parent());
		}
	};

private:
	Tparent* _parent;
	children_t _children;
};

/*! \brief Special class to represent the fact that a tree object is at the root of the hierarchy.
*/
template <typename Tchild>
struct PVDataTreeNoParent
{
	typedef PVDataTreeNoParent parent_t;
	inline PVDataTreeNoParent* get_parent()
	{
		std::cout << "WARNING: The data tree object has no ancestor of such specified type" << std::endl;
		assert(false);
		return NULL;
	}
	void add_child(Tchild* /*child*/) {}
	void remove_child(Tchild* /*child*/){}
};

/*! \brief Special class to represent the fact that a tree object is not meant to have any children.
*/
template <typename Tparent>
struct PVDataTreeNoChildren
{
	typedef Tparent parent_t;
	void dump(uint32_t /*spacing*/){}
};
}

#endif /* PVDATATREEOBJECT_H_ */
