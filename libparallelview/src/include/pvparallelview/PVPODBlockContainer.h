
#define BLOCKCOINTAINER_H

#define DEFAULT_BLOCK_SIZE_BYTES 5*1024*1024

#include <list>
#include <utility>
#include <iostream>
#include <iterator>
#include <cassert>

#include <pvkernel/core/type_select.h>

#include <omp.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace PVCore {

template<typename T, typename size_type_, template Alloc = std::allocator<T> >
class PVPODBlockContainer: protected Alloc
{
	// Assert that Alloc is an allocator for T
	BOOST_STATIC_ASSERT((boost::is_same<typename Alloc::value_type, T>::value));
public:
	typedef Alloc allocator_type;
	typedef size_type_ size_type;
	typedef T value_type;
	typedef typename allocator_type::pointer pointer;
	typedef typename allocator_type::reference reference;
	typedef typename allocator_type::const_pointer const_pointer;
	typedef typename allocator_type::const_reference const_reference;

private:
	struct block_t
	{
		pointer p;
	};
	typedef std::list<block_t> list_blocks_t;
	typedef typename list_blocks_t::iterator list_blocks_iterator_t;
	typedef typename list_blocks_t::const_iterator list_blocks_const_iterator_t;

public:
#if 0
	template <bool is_const>
	class PODBlockContainerIterator: public std::iterator<std::forward_iterator_tag, value_type>
	{
		typedef typename type_select<is_const, const_pointer, pointer>::r ptr_t;
		typedef typename type_select<is_const, const_reference, reference>::r ref_t;
		typedef typename type_select<is_const, list_blocks_const_iterator_t, list_blocks_iterator_t>::r it_block_t;
		typedef PODBlockContainerIterator _it_type;
	public:
		PODBlockContainerIterator(): _p(NULL), _lb(NULL) {}
		PODBlockContainerIterator(ptr_t p, ptr_t end_p, it_block_t b, list_blocks_t const& lb): _p(p), _end_p(end_p), _block_it(b), _lb(&lb) {}
	public:
		bool operator==(const _it_type& other) { assert(_p); return _p == other._p; }
		bool operator!=(const _it_type& other) { assert(_p); return _p != other._p; }

	public:
		_it_type& operator++()
		{
			assert(_p);
			_p++;
			if (_p != _end_p && (_p >= _block_it->first + _block_it->second)) {
				_block_it++;
				_p = _block_it->first;
			}
			return *this;
		}
		const _it_type operator++(int) { _it_type it_ret(*this); operator++(); return it_ret; }
		// TODO: finish this
		it_type& operator+(int n)
		{
			it_ptr newp = _p + n;
			if (newp >= _block_it->first + _block_it->second) { // If it goes beyond the current block
				// Compute current position in the block
				uintptr_t pos = (uintptr_t)_p - (uintptr_t)_block_it->first;
				_block_it++;
				if (_block_it == _lb.end()) {
					block_t const& last_block = _lb.back();
					_p = last_block.first + last_block.second; // _p == container::end()
					return *this;
				}
			}
		}

	public:
		ptr_t operator->() { assert(_p); return _p; }
		ref_t operator*() { assert(_p); return *_p; }


	private:
		ptr_t _p;
		ptr_t _end_p;
		it_block_t _block_it;
		list_blocks_t const* _lb;
	};

	typedef PODBlockContainerIterator<true> const_iterator;
	typedef PODBlockContainerIterator<false> iterator;
#endif

public:
	PVPODBlockContainer(size_type block_size, allocator_type const& a = allocator_type()) :
		allocator_type(a)
	{
		init();
		set_block_size(block_size);
	}

	~PVPODBlockContainer()
	{
		free();
	}
private:
	void init()
	{
	}
public:
	size_type size() const { return (_cur_block-_block.begin())*_size_block + _cur_block_size; }
	size_type capacity() const { return _size_block*_block.size(); }
	size_type max_size() const { return allocator_type::max_size(); }

	void reserve_blocks(

#if 0
	iterator begin() { return iterator(_blocks.front().first, _cur_elt, _blocks.begin(), _blocks); }
	const_iterator begin() const { return const_iterator(_blocks.front().first, _cur_elt, _blocks.begin(), _blocks); }

	iterator end() { return iterator(_cur_elt, _cur_elt, _cur_block, _blocks); }
	const_iterator end() const { return const_iterator(_cur_elt, _cur_elt, _cur_block, _blocks); }

	reference front() { return *(_blocks.front().first); }
	const_reference front() const { return *(_blocks.front().first); }

	reference back() { return *(_cur_elt-1); }
	const_reference back() const { return *(_cur_elt-1); }

	void free()
	{
		list_blocks_const_iterator_t it;
		for (it = _blocks.begin(); it != _blocks.end(); it++) {
			pointer p = it->first;
#pragma omp parallel for
			for (size_type i = 0; i < it->second; i++) {
				(&p[i])->~T();
			}
			_alloc.deallocate(it->first, it->second);
		}
		init();
	}

private:
	list_blocks_iterator_t _allocate_new_block(size_type n)
	{
		std::cerr << "Allocate " << n << " elements." << std::endl;
		pointer new_block = _alloc.allocate(n);
#pragma omp parallel for
		for (size_type i = 0; i < n; i++) {
			new (&new_block[i]) T(_def_v);
		}
		_blocks.push_back(block_t(new_block, n));
		_total_elts += n;
		list_blocks_iterator_t ret = _blocks.end(); ret--;
		return ret;
	}

	pointer _get_next_elt()
	{
		pointer ret;
		if (_cur_block_free_elts > 0) {
			_cur_block_free_elts--;
			ret = _cur_elt;
			_cur_elt++;
			// TODO: pre-allocation in a thread of the next block if this one starts to become smaller ?
		}
		else {
			size_type size_new_block;
			if (_estimate_max_elts > _total_elts) {
				size_new_block = _estimate_max_elts - _total_elts;
			}
			else {
				size_new_block = _default_block_size;
			}
			_set_cur_block(_allocate_new_block(size_new_block));
			_cur_block_free_elts--;
			ret = _cur_elt;
			_cur_elt++;
		}
		_size++;
		return ret;
	}

	void _set_cur_block(list_blocks_iterator_t it)
	{
		// This assumes that the current block has never been used.
		_cur_block = it;
		_cur_block_free_elts = it->second;
		_cur_elt = it->first;
	}
#endif

private:
	list_blocks_t _blocks;
	list_blocks_iterator_t _cur_block;

	size_type _cur_block_size;
	size_type _block_size;
};


}

#endif
