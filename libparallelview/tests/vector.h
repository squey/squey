#ifndef OWN_VECTOR_H
#define OWN_VECTOR_H

#include <algorithm>

#define MAX_SIZE 10000

/*****************************************************************************
 * version "classique"
 */

template <class C, int SIZE = 1000, int INCREMENT = 1000>
class Vector1
{
public:
	Vector1(const unsigned size = SIZE) :
		_size(size),
		_index(0)
	{
		_array = 0;
	}

	~Vector1()
	{
		clear();
	}

	void reserve(const unsigned size)
	{
		reallocate(size);
	}

	void reset()
	{
		_index = 0;
	}

	inline unsigned size() const
	{
		return _index;
	}

	inline size_t memory() const
	{
		if(_array != 0) {
			return sizeof(Vector1) + _size * sizeof(C);
		} else {
			return sizeof(Vector1);
		}
	}

	inline bool is_null() const
	{
		return (_array == 0);
	}

	void clear()
	{
		if(_array) {
			free(_array);
			_array = 0;
			_index = 0;
		}
	}

	inline C &at(const int i)
	{
		return _array[i];
	}

	inline C const& at(const int i) const
	{
		return _array[i];
	}

	inline void push_back(const C &c)
	{
		if (_index == _size) {
			reallocate(_size + INCREMENT);
		}
		_array[_index++] = c;
	}

	C &back() const
	{
		return _array[_index-1];
	}

	Vector1<C> &operator=(const Vector1<C> &v)
	{
		clear();
		if(v._size) {
			_index = v._index;
			reserve(v._size);
			std::copy(v._array, v._array + _index, _array);
		}
		return *this;
	}

	bool operator==(const Vector1<C> &v) const
	{
		if(_index != v._index) {
			return false;
		}

		if(_array == 0) {
			if (v._array == 0) {
				return true;
			} else {
				return false;
			}
		} else if(v._array == 0) {
			return true;
		} else {
			return std::equal(_array, _array + _index, v._array);

		}
	}

private:
	void reallocate(const unsigned size)
	{
		_array = (C*) realloc(_array, (size) * sizeof(C));
		_size = size;
	}

private:
	C        *_array;
	unsigned  _size;
	unsigned  _index;
};


/* un truc utile pour Vector2 et Vector3
 */
enum {
	VectorImplIndex = 0,
	VectorImplSize
};

/*****************************************************************************
 * version avec classe Impl
 */

template <class C>
class Vector2Impl
{
public:
	static Vector2Impl *allocate(unsigned size)
	{
		Vector2Impl *p = (Vector2Impl*) malloc((size * sizeof(C)) + sizeof(Vector2Impl));
		p->size = size;
		p->index = 0;
		return p;
	}

	C *buffer() const
	{
		return (C*)(this + 1);
	}

	static Vector2Impl *reallocate(Vector2Impl *p, const unsigned size)
	{
		p = (Vector2Impl *)realloc(p, (size) * sizeof(C) + sizeof(Vector2Impl));
		p->size = size;
		return p;
	}

public:
	unsigned size;
	unsigned index;
};

template <class C, int SIZE = 1000, int INCREMENT = 1000>
class Vector2
{
public:
	Vector2()
	{
		_v = 0;
	}

	~Vector2()
	{
		clear();
	}

	void clear()
	{
		if(_v != 0) {
			free(_v);
			_v = 0;
		}
	}

	unsigned size() const
	{
		return _v->index;
	}

	void reserve(const unsigned size)
	{
		_v = Vector2Impl<C>::allocate(size);
		_v->index = 0;
	}

	inline bool is_null()
	{
		return (_v == 0);
	}

	inline C &at(const int i)
	{
		return _v->buffer()[i];
	}

	inline C const& at(const int i) const
	{
		return _v->buffer()[i];
	}

	void push_back(const C &c)
	{
		if (_v == 0) {
			_v = Vector2Impl<C>::allocate(SIZE);
		} else if(_v->size ==_v->index) {
			_v = Vector2Impl<C>::reallocate(_v, _v->size + INCREMENT);
		}
		_v->buffer()[_v->index] = c;
		++(_v->index);
	}

	C &back() const
	{
		return _v->buffer()[_v->index-1];
	}

private:
	Vector2Impl<C> *_v;
};

#endif // OWN_VECTOR_H
