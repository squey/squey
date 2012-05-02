#ifndef OWN_VECTOR_H
#define OWN_VECTOR_H

enum {
	SW = 0,
	SE,
	NW,
	NE
};

#define MAX_SIZE 10000

template <class C>
class Vector
{
public:
	Vector(unsigned size = 1000, unsigned increment = 1000) :
		_size(size),
		_increment(increment),
		_index(0)
	{
		_array = 0;
	}

	~Vector()
	{
		clear();
	}

	void reserve(unsigned size)
	{
		reallocate(size);
	}

	unsigned size()
	{
		return _index;
	}

	bool is_null()
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

	C &at(int i)
	{
		return _array[i];
	}

	void push_back(C &c)
	{
		if ((_index == _size) || (_array == 0)) {
			reallocate(_size + _increment);
		}
		_array[_index++] = c;
	}

private:
	void reallocate(unsigned size)
	{
		_array = (C*) realloc(_array, (size) * sizeof(C));
		_size = size;
	}

private:
	C        *_array;
	unsigned  _size;
	unsigned  _increment;
	unsigned  _index;
};

#endif // OWN_VECTOR_H
