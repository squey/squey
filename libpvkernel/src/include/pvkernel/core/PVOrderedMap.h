/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVCORE_PVORDEREDMAP_H
#define PVCORE_PVORDEREDMAP_H

#include <stdexcept>
#include <vector>
#include <algorithm>

namespace PVCore
{

/**
 * \class PVOrderedMapNode
 * \brief A node<key,value> for a PVOrderedMap.
 *
 * Inspired from https://github.com/zzjin/CoopES/blob/master/include/ssyntax/qorderedmap.h
 */
template <class Key, class Value>
class PVOrderedMapNode
{
  public:
	/**
	 * \brief ordered map constructor
	 * \param key the new key
	 * \param value the value of key
	 *
	 * Create and initialize the ordered map node's object with the given parameters.
	 */
	PVOrderedMapNode(Key key, Value value) : _key(std::move(key)), _value(std::move(value)) {}

	/**
	 * \brief return the map node's key .
	 * \return key
	 *
	 * Return the map node's key.
	 */
	const Key& key() const { return _key; }

	/**
	 * \brief return the map node's value .
	 * \return value
	 *
	 * Return the map node's value.
	 */
	const Value& value() const { return _value; }

	/**
	 * \brief return the map node's value .
	 * \return value
	 *
	 * Return the map node's value.
	 */
	Value& value() { return _value; }

  private:
	Key _key;     /*!< node's key */
	Value _value; /*!< node's value */
};

/**
 * \class PVOrderedMap
 * \brief An ordered Map implementation
 *
 * In this map, the objects are ordered inside at the same order
 * they were inserted.
 *
 */
template <class Key, class Value>
class PVOrderedMap
{
  public:
	/**
	 * \brief default construstor
	 *
	 * Creates and initialized the ordered map object.
	 */
	PVOrderedMap() = default;

	/**
	 * \brief Removes all elements from the container.
	 *
	 * Removes all elements from the container.
	 * Use std:vector::clear()
	 */
	void clear();

	/**
	 * \brief Check for the key in the map
	 * \param key the key to be tested
	 * \return true if the key is found on the map, false otherwise
	 *
	 * This function checks if the key is found in the ordered map.
	 */
	bool contains(const Key& key) const;

	/**
	 * \brief Check if the map is empty
	 * \return true if the map is empty
	 *
	 * Returns true if there are no keys on the map.
	 */
	bool empty() const;

	/**
	 * \brief Returns the number of elements in the container.
	 * \return The number of elements in the container.
	 *
	 * This function returns the number of elements in the container.
	 */
	size_t size() const;

	/**
	 * \brief Returns a reference of a list containing all the keys in the container at insert time
	 *order.
	 * \return The reference of a list containing all the keys in the container.
	 *
	 * Returns a reference of a list containing all the keys in the container at insert time order.
	 */
	const std::vector<Key> keys() const;

	/*!< Typedef for Key. Provided for STL compatibility. */
	using key_type = Key;

	/*!< Use the std::vector::const_iterator */
	using const_iterator = typename std::vector<PVOrderedMapNode<Key, Value>>::const_iterator;

	/*!< Use the std::vector::iterator */
	using iterator = typename std::vector<PVOrderedMapNode<Key, Value>>::iterator;

	/**
	 * \brief Returns The std::vector::begin() iterator.
	 * \brief Returns The std::vector::begin() iterator.
	 */
	iterator begin();

	/**
	 * \brief Returns The std::vector::begin() iterator.
	 * \brief Returns The std::vector::begin() iterator.
	 */
	const_iterator begin() const;

	/**
	 * \brief Returns The std::vector::end() iterator.
	 * \brief Returns The std::vector::end() iterator.
	 */
	const_iterator end() const;

	/**
	 * \brief Returns The std::vector::end() iterator.
	 * \brief Returns The std::vector::end() iterator.
	 */
	iterator end();

	/**
	 * \brief Finds an element with key equivalent to key.
	 * \param key The key of the element to search for.
	 * \return Iterator to an element with key equivalent to key.
	 *
	 * This function returns the iterator to an element with key equivalent to key.
	 * If no such element is found, the end() iterator is returned.
	 */
	const_iterator find(const Key& key) const;

	/**
	 * \brief Finds an element with key equivalent to key.
	 * \param key The key of the element to search for.
	 * \return Iterator to an element with key equivalent to key.
	 *
	 * This function returns the iterator to an element with key equivalent to key.
	 * If no such element is found, the end() iterator is returned.
	 */
	iterator find(const Key& key);

	/**
	 * \brief Return a reference to the mapped value of the key.
	 * \param key The key of the element to find.
	 * \return A reference to the mapped value of the key.
	 *
	 * This function returns a reference to the mapped value of the key if the key exists.
	 * Otherwise a reference to the mapped value of the new element create from this key.
	 */
	Value& operator[](const Key& key);

	/**
	 * \brief Return a reference to the mapped value of the key.
	 * \param key The key of the element to find.
	 * \return A reference to the mapped value of the key.
	 *
	 * Returns a reference to the mapped value of the element with key equivalent to key.
	 * If no such element exists, an exception of type std::out_of_range is thrown.
	 */
	const Value& at(const Key& key) const;

	/**
	 * \brief Return a reference to the mapped value of the key.
	 * \param key The key of the element to find.
	 * \return A reference to the mapped value of the key.
	 *
	 * Returns a reference to the mapped value of the element with key equivalent to key.
	 * If no such element exists, an exception of type std::out_of_range is thrown.
	 */
	Value& at(const Key& key);

  private:
	std::vector<PVOrderedMapNode<Key, Value>> _nodes; /*!< the list of nodes */
};

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::clear
 *
 *****************************************************************************/
template <class Key, class Value>
void PVOrderedMap<Key, Value>::clear()
{
	_nodes.clear();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::contains
 *
 *****************************************************************************/
template <class Key, class Value>
bool PVOrderedMap<Key, Value>::contains(const Key& key) const
{
	return std::any_of(_nodes.begin(), _nodes.end(),
	                   [&key](PVOrderedMapNode<Key, Value> const& n) { return n.key() == key; });
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::empty
 *
 *****************************************************************************/
template <class Key, class Value>
bool PVOrderedMap<Key, Value>::empty() const
{
	return _nodes.empty();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::size
 *
 *****************************************************************************/
template <class Key, class Value>
size_t PVOrderedMap<Key, Value>::size() const
{
	return _nodes.size();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::keys
 *
 *****************************************************************************/
template <class Key, class Value>
const std::vector<Key> PVOrderedMap<Key, Value>::keys() const
{
	std::vector<Key> list;

	for (PVOrderedMapNode<Key, Value> const& n : _nodes) {
		list.push_back(n.key());
	}

	return list;
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::begin
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::const_iterator PVOrderedMap<Key, Value>::begin() const
{
	return _nodes.begin();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::begin
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::iterator PVOrderedMap<Key, Value>::begin()
{
	return _nodes.begin();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::end
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::const_iterator PVOrderedMap<Key, Value>::end() const
{
	return _nodes.end();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::end
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::iterator PVOrderedMap<Key, Value>::end()
{
	return _nodes.end();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::find
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::const_iterator
PVOrderedMap<Key, Value>::find(const Key& key) const
{
	return std::find_if(_nodes.begin(), _nodes.end(),
	                    [&key](PVOrderedMapNode<Key, Value> const& n) { return n.key() == key; });
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::find
 *
 *****************************************************************************/
template <class Key, class Value>
typename PVOrderedMap<Key, Value>::iterator PVOrderedMap<Key, Value>::find(const Key& key)
{
	return std::find_if(_nodes.begin(), _nodes.end(),
	                    [&key](PVOrderedMapNode<Key, Value> const& n) { return n.key() == key; });
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::operator[]
 *
 *****************************************************************************/
template <class Key, class Value>
Value& PVOrderedMap<Key, Value>::operator[](const Key& key)
{
	// test if key exist, only return its value
	auto it = find(key);
	if (it != _nodes.end()) {
		return it->value();
	}

	// key doesn't exist, we add a new node and return its value
	_nodes.push_back({key, Value()});
	return _nodes.back().value();
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::at
 *
 *****************************************************************************/
template <class Key, class Value>
Value& PVOrderedMap<Key, Value>::at(const Key& key)
{
	// test if key exist, only return its value
	auto it = find(key);
	if (it != _nodes.end()) {
		return it->value();
	}

	// key doesn't exist
	throw std::out_of_range("This key doesn't exist in this PVOrderedMap");
}

/******************************************************************************
 *
 * PVCore::PVOrderedMap<Key, Value>::at
 *
 *****************************************************************************/
template <class Key, class Value>
const Value& PVOrderedMap<Key, Value>::at(const Key& key) const
{
	// test if key exist, only return its value
	auto it = find(key);
	if (it != _nodes.end()) {
		return it->value();
	}

	// key doesn't exist
	throw std::out_of_range("This key doesn't exist in this PVOrderedMap");
}

} // namespace PVCore

#endif // PVCORE_PVORDEREDMAP_H
