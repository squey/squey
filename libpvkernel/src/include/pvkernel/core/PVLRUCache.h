/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#ifndef PVCORE_PVLRUCACHE_H
#define PVCORE_PVLRUCACHE_H

#include <unordered_map>
#include <list>

#include <cassert>

namespace PVCore
{

/**
 * A template LRU (Least Recently Used) cache class.
 *
 * @note almost copied from http://stackoverflow.com/questions/2504178/lru-cache-design
 */
template <typename K, typename V>
class PVLRUCache
{
  public:
	/**
	 * Constructor
	 */
	PVLRUCache(size_t size) : _size(size) {}

  public:
	/**
	 * Change the maximum number of cached entries
	 *
	 * @param size the new size
	 */
	void resize(size_t size)
	{
		assert(size > 0);

		_size = size;
		clean();
	}

  public:
	/**
	 * Check for key existence
	 *
	 * @param k the key to test for existence
	 *
	 * @return true if k exists; false otherwise
	 */
	bool exist(const K& k) const { return _map.count(k) != 0; }

	/**
	 * Invalide all cached entries
	 */
	void invalidate()
	{
		_map.clear();
		_list.clear();
	}

  public:
	/**
	 * Insert a new entry
	 *
	 * @param k the entry key
	 * @param v the entry value
	 */
	void insert(const K& k, const V& v)
	{
		auto it = _map.find(k);

		if (it != _map.end()) {
			_list.erase(it->second);
			_map.erase(it);
		}

		_list.push_front(std::make_pair(k, v));

		clean();
	}

	/**
	 * Retrieve a value given its key
	 *
	 * @param k the entry key
	 *
	 * @return entry value corresponding to k
	 *
	 * @note k must be a valid entry key, see exist
	 */
	const V& get(const K& k)
	{
		assert(exist(k));

		auto it = _map.find(k);

		_list.splice(_list.begin(), _list, it->second);

		return it->second->second;
	}

  private:
	void clean()
	{
		while (_map.size() > _size) {
			auto it = _list.end();
			--it;
			_map.erase(it->first);
			_list.pop_back();
		}
	}

  private:
	std::list<std::pair<K, V>> _list;
	std::unordered_map<K, decltype(_list.begin())> _map;
	size_t _size;
};

} // namespace PVCore

#endif // PVCORE_PVLRUCACHE_H
