#ifndef PVCORE_PVPODTREE_H
#define PVCORE_PVPODTREE_H

#include <pvkernel/core/general.h>

#include <boost/math/common_factor_rt.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <iterator>

#include <cmath>
#include <cassert>
#include <iostream>

namespace PVCore {

template <typename T, typename size_type_ = size_t, size_type_ NB = 1000, class Alloc = std::allocator<T> >
class PVPODTree: protected Alloc::template rebind<unsigned char>::other
{
	BOOST_STATIC_ASSERT((boost::is_pod<T>::value));
	BOOST_STATIC_ASSERT((boost::is_same<typename Alloc::value_type, T>::value));
public:
	typedef Alloc allocator_type;
	typedef typename allocator_type::template rebind<unsigned char>::other allocator_byte;
	typedef size_type_ size_type;
	typedef T value_type;
	typedef typename allocator_type::pointer pointer;
	typedef typename allocator_type::reference reference;
	typedef typename allocator_type::const_pointer const_pointer;
	typedef typename allocator_type::const_reference const_reference;
	typedef PVPODTree<value_type, size_type, NB, allocator_type> this_type;

private:
	struct block_t
	{
		block_t(): p((pointer)-1) { }

		pointer p;
		inline block_t* next(size_type nelts_block) const { return ((block_t*) (p+nelts_block)); }
	};
	struct branch_t
	{
		block_t first;
		block_t cur;
		pointer p_cur_block;

		inline bool valid() const { return first.p != (pointer)-1; }
		inline void init_first(block_t b) { first = b; cur = first; p_cur_block = b.p; }
		inline void set_next(block_t b, size_type nelts_block)
		{
			*(cur.next(nelts_block)) = b;
			cur = b;
			p_cur_block = cur.p;
		}
		inline void push(T v) { *p_cur_block = v; p_cur_block++; }
		// Returns the number of blocks, even if the current one isn't complete.
		inline size_type nb_blocks(size_type nelts_block) const
		{
			if (!valid()) {
				return 0;
			}
			block_t bvisit = first;
			size_type ret = 1;
			while (bvisit.p != cur.p) {
				bvisit = *bvisit.next(nelts_block);
				ret++;
			}
			return ret;
		}

		inline size_type nb_elts(size_type nelts_block) const
		{
			size_type ret = nb_blocks(nelts_block);
			if (ret == 0) {
				return 0;
			}
			ret = (ret-1)*nelts_block + cur_size();
			return ret;
		}

		inline size_type cur_size() const { return p_cur_block-cur.p; }
		inline T const& get_first() const { assert(valid()); return *(first.p); }

		inline void move_branch(branch_t& other, size_type nelts_block)
		{
			if (other.valid() && other.cur_size() > 0) {
				if (valid()) {
					*(cur.next(nelts_block)) = other.first;
					cur = other.cur;
					p_cur_block = other.p_cur_block;
				}
				else {
					*this = other;
				}
				other.first.p = NULL;
			}
		}
	};

public:
	class const_branch_iterator: public std::iterator<std::forward_iterator_tag, value_type>
	{
		friend class PVPODTree<value_type, size_type, NB, allocator_type>;
	protected:
		const_branch_iterator(branch_t const& branch, const block_t& cur_block, size_type block_size, size_type cur_index = 0):
			_branch(&branch), _cur_block(cur_block), _block_size(block_size), _cur_index(cur_index)
		{ }
	public:
		const_branch_iterator():
			_branch(NULL), _cur_index(0)
		{ }

		T const& operator*() const { return get_cur_ref(); }

		const_branch_iterator& operator++()
		{
			assert(_cur_block.p && _branch);
			if (_cur_index == _block_size-1) {
				_cur_index = 0;
				_cur_block = *(_cur_block.next(_block_size));
			}
			else {
				_cur_index++;
				if (_cur_block.p == _branch->cur.p &&
				    _cur_index == _branch->cur_size()) {
					// We have reached the end of this branch.
					_cur_block = block_t();
					_cur_index = 0;
				}
			}
			if (_cur_block.p != (pointer)-1 && get_cur_ref() == (value_type)(-1)) {
				// This block is half-full (result of a merging). Process to the next one.
				_cur_block = *(_cur_block.next(_block_size));
				_cur_index = 0;
			}
			return *this;
		}

		inline const_branch_iterator& operator++(int)
		{
			return operator++();
		}
	
	public:
		bool operator==(const const_branch_iterator& o) { return _cur_block.p == o._cur_block.p &&
		                                                         _cur_index == o._cur_index; }
		bool operator!=(const const_branch_iterator& o) { return _cur_block.p != o._cur_block.p ||
		                                                         _cur_index != o._cur_index; }

	private:
		inline T const& get_cur_ref() const { assert(_cur_block.p); return *(_cur_block.p + _cur_index); }
	private:
		branch_t const* _branch;
		block_t _cur_block;
		size_type _block_size;
		size_type _cur_index;
	};
public:
	PVPODTree():
		_buf(NULL)
	{
		clear();
	}

	PVPODTree(size_type nelts):
		_buf(NULL)
	{
		clear();
		resize(nelts);
	}

	~PVPODTree()
	{
		clear();
	}

public:
	void clear()
	{
		if (_buf) {
			allocator_byte::deallocate((unsigned char*) _buf, 0);
			_buf = NULL;
		}
		_cur_buf = NULL;
#ifndef NDEBUG
		_buf_size = 0;
#endif
		_nblocks_max = 0;
		_nelts_block = 0;
		typename std::list<pointer>::const_iterator it;
		for (it = _extra_bufs.begin(); it != _extra_bufs.end(); it++) {
			allocator_byte::deallocate((unsigned char*) *it, 0);
		}
	}

	void resize(size_type nelts)
	{
		// Compute buf size such as only one allocation is needed.
		// The buffer will be organized this way :
		// [[T] [T] [T] ... [T] [pointer to next block] [T] [T] ... [T] [pointer to next block]]
		// [<-----------------------------------------> <------------------------------------->]
		// [      block 0                                   block 1                        ... ]
		// Each branch take a block from this buffer and set its pointer to the next block when necessary.
		// By doing this, we save memory with no penality on performances !

		clear();

		size_type nblocks_max;
		if (nelts >= NB) {
			_nelts_block = (nelts+NB-1)/NB;
			nblocks_max = (nelts+_nelts_block-1)/(_nelts_block) + NB/2;
		}
		else {
			//_nelts_block = picviz_max(10, std::sqrt(nelts));
			_nelts_block = 10;
			//nblocks_max = NB*(((nelts/_nelts_block)+NB-1)/NB) + nelts%(NB);
			//nblocks_max = (nelts+_nelts_block+1)/_nelts_block + NB/2;
			nblocks_max = nelts;
		}
		//assert(nblocks_max >= NB);

		//PVLOG_INFO("(PVPODTree::resize): number of elts in a block = %llu\n", _nelts_block);
		//PVLOG_INFO("(PVPODTree::resize): maximum number of blocks = %llu\n", nblocks_max);
		_nblocks_max = nblocks_max;

		// Then, compute the size of the buffer
		size_t buf_size = _nblocks_max*(size_block());
		_buf = (pointer) allocator_byte::allocate(buf_size);
		assert(_buf);
		memset(_buf, 0xFF, buf_size);
		_cur_buf = _buf;
#ifndef NDEBUG
		_buf_size = buf_size;
#endif
		
		// Init tree
		/*for (size_type i = 0; i < NB; i++) {
			_tree[i].init_first(reserve_block());
		}*/
	}

	bool push(size_type branch_id, T elt)
	{
		bool first = false;
		assert(branch_id < NB);
		branch_t& cur_b(_tree[branch_id]);
		if (!cur_b.valid()) {
			first = true;
			cur_b.init_first(reserve_block());
		}
		else
		if (cur_b.cur_size() == _nelts_block) {
			cur_b.set_next(reserve_block(), _nelts_block);
		}
		cur_b.push(elt);
		return first;
	}

	size_type number_blocks_used() const
	{
		return ((uintptr_t)_cur_buf - (uintptr_t)_buf)/(_nelts_block*sizeof(T) + sizeof(pointer));
	}

	inline T const& get_first_elt_of_branch(size_type branch_id) const
	{
		assert(branch_id < NB);
		return _tree[branch_id].get_first();
	}

	inline bool branch_valid(size_type branch_id) const
	{
		assert(branch_id < NB);
		return _tree[branch_id].valid();
	}

	inline size_type branch_nelts(size_type branch_id) const
	{
		assert(branch_id < NB);
		return _tree[branch_id].nb_elts(_nelts_block);
	}

	inline void move_branch(size_type branch_id, size_type other_branch_id, this_type& other)
	{
		assert(branch_id < NB);
		assert(other_branch_id < NB);
		assert(_nelts_block == other._nelts_block);
		_tree[branch_id].move_branch(other._tree[other_branch_id], _nelts_block);
	}

	inline void take_buf(this_type& other)
	{
		_extra_bufs.push_back(other._buf);
		other._buf = NULL;
		other.clear();
	}

	inline const_branch_iterator begin_branch(size_type branch_id) const
	{
		assert(branch_id < NB);
		return const_branch_iterator(_tree[branch_id], _tree[branch_id].first, _nelts_block);
	}

	inline const_branch_iterator end_branch(size_type /*branch_id*/) const
	{
		return const_branch_iterator();
	}

	static inline size_type nbranches() { return NB; }

	void dump_buf_stats() const
	{
		const size_t usage = ((uintptr_t) _cur_buf - (uintptr_t) _buf);
		const size_t org_size = size_block()*_nblocks_max;
		PVLOG_INFO("Buffer usage: %0.4fMB/%0.4fMB (%0.4f %%)\n", ((double)usage/((1024.0*1024.0))), ((double)org_size)/(1024.0*1024.0), ((double)usage/(double)org_size)*100.0);
	}

	void dump_branch_stats() const
	{
		for (size_type b = 0; b < NB; b++) {
			std::cerr << b << "," << branch_nelts(b) << std::endl;
		}
	}

private:
	inline size_t size_block() const { return (size_t)_nelts_block*sizeof(value_type)+sizeof(branch_t); }
	inline block_t reserve_block()
	{
		block_t ret;
		ret.p = _cur_buf;
		_cur_buf = (pointer) (((unsigned char*)_cur_buf) + size_block());
		assert(_cur_buf <= (pointer) ((unsigned char*) _buf + _buf_size));
		return ret;
	}

	branch_t _tree[NB];
	pointer _buf;
	std::list<pointer> _extra_bufs;
#ifndef NDEBUG
	size_t _buf_size;
#endif
	pointer _cur_buf;
	size_type _nblocks_max;
	size_type _nelts_block;
};

}


#endif
