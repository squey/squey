//! \file PVMatrix.h
//! $Id: PVMatrix.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PVCORE_PVMATRIX_H
#define PVCORE_PVMATRIX_H

#include <pvkernel/core/stdint.h>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/bind.hpp>
#include <vector>

#include <pvkernel/core/PVTypeTraits.h>

namespace PVCore {

void __transpose_float(float* res, float* data, uint32_t nrows, uint32_t ncols);

template <class T, class IndexRow, class IndexCol, bool is_float_multiple = PVTypeTraits::is_size_multiple<T, float>::value, bool bigger_than = PVTypeTraits::bigger_than<T, float>::value, bool is_pod = boost::is_pod<T>::value >
struct PVMatrixComputation
{
	static void transpose(T* res, T* data, IndexRow nrows, IndexCol ncols)
	{
		// TODO: optimise this !
		for (IndexRow i = 0; i < nrows; i++) {
			for (IndexCol j = 0; j < ncols; j++) {
				res[j*nrows + i] = data[i*ncols + j];
			}
		}
	}

	static void set_rows_value(T* data, IndexRow a, IndexRow b, IndexCol ncols, T const& v)
	{
	#pragma omp parallel for
		for (ssize_t i = (ssize_t) a*ncols; i < (ssize_t) (b+1)*ncols; i++) {
			new (&data[i]) T(v);
		}
	}
};

// If `T' is a POD, a multiple of sizeof(float) less or equal than sizeof(float), then we can optimise its transposition as it is done with floatting values with SSE.
template <class T>
struct PVMatrixComputation<T, uint32_t, uint32_t, true, false, true>
{
	static void transpose(T* res, T* data, uint32_t nrows, uint32_t ncols)
	{
		__transpose_float((float*) res, (float*) data, nrows*sizeof(float)/sizeof(T), ncols*sizeof(float)/sizeof(T));
	}

	static void set_rows_value(T* data, uint32_t a, uint32_t b, uint32_t ncols, T v)
	{
		for (ssize_t i = (ssize_t) a*ncols; i < (ssize_t) (b+1)*ncols; i++) {
			data[i] = v;
		}
	}
};

template <class T, class IndexRow, class IndexCol, bool mod, bool greater>
struct PVMatrixComputation<T, IndexRow, IndexCol, mod, greater, true>
{
	static void transpose(T* res, T* data, IndexRow nrows, IndexCol ncols)
	{
		// TODO: optimise this !
		for (IndexRow i = 0; i < nrows; i++) {
			for (IndexCol j = 0; j < ncols; j++) {
				res[j*nrows + i] = data[i*ncols + j];
			}
		}
	}

	static void set_rows_value(T* data, IndexRow a, IndexRow b, IndexCol ncols, T const& v)
	{
		for (size_t i = (size_t) a*ncols; i < (size_t) (b+1)*ncols; i++) {
			data[i] = v;
		}
	}
};

template <typename T, typename IndexRow = uint32_t, typename IndexCol = uint32_t, template <class Talloc> class Alloc = std::allocator>
class PVMatrix
{
	typedef Alloc<T> allocator_type;
	typedef T value_type;
	typedef boost::is_pod<value_type> value_pod;
	typedef typename allocator_type::pointer pointer;
	typedef typename allocator_type::const_pointer const_pointer;
	typedef typename allocator_type::reference reference;
	typedef typename allocator_type::const_reference const_reference;
	typedef IndexCol index_col;
	typedef IndexRow index_row;
	typedef PVMatrix<T, IndexRow, IndexCol, Alloc> matrix_type;
	// Define the transposed matrix type
public:
	typedef PVMatrix<T, index_col, index_row, Alloc> transposed_type;
	friend class PVMatrix<T, index_col, index_row, Alloc>;

public:
	class PVMatrixLine
	{
	public:
		PVMatrixLine(PVMatrix<T, IndexRow, IndexCol, Alloc> const& parent, index_row row)
		{
			_p = parent.get_row_ptr(row);
			_size = parent.get_width();
		}
	public:
		inline index_col size() const { return _size; }
		inline index_col count() const { return _size; }

		inline reference at(index_col col) { return _p[col]; }
		inline const_reference at(index_col col) const { return _p[col]; }
		inline reference operator[](index_col col) { return _p[col]; }
		inline const_reference operator[](index_col col) const { return _p[col]; }

		inline void set_value(index_col col, const_reference v) { assert(col < _size); _p[col] = v; }

	private:
		pointer _p;
		index_col _size;
	};

	class PVMatrixLineConst
	{
	public:
		PVMatrixLineConst(PVMatrix<T, IndexRow, IndexCol, Alloc> const& parent, index_row row)
		{
			_p = parent.get_row_ptr(row);
			_size = parent.get_width();
		}
	public:
		inline index_col size() const { return _size; }
		inline index_col count() const { return _size; }

		inline const_reference at(index_col col) const { return _p[col]; }
		inline const_reference operator[](index_col col) const { return _p[col]; }

	private:
		const_pointer _p;
		index_col _size;
	};
public:
	typedef PVMatrixLineConst const_line;
	typedef PVMatrixLine line;

public:
	PVMatrix(allocator_type const& a = allocator_type()): _ncols(0), _nrows(0), _nrows_physical(0), _data(NULL), _alloc(a) {}
	PVMatrix(index_row nrows, index_col ncols, allocator_type const& a = allocator_type()):
		_data(NULL), _alloc(a)
   	{ reserve(nrows, ncols); }

	virtual ~PVMatrix() { free(); }
private:
	PVMatrix(const PVMatrix& org) { assert(false); }
	PVMatrix& operator=(const PVMatrix& org) { assert(false); return *this; }
public:
	void clear()
	{
		if (_data) {
			resize_nrows(0);
		}
	}

	bool resize(index_row nrows, index_col ncols, const_reference v = value_type())
	{
		if (_ncols > 0 && ncols == _ncols) {
			return resize_nrows(nrows);
		}

		if (_nrows > 0 && _nrows == nrows) {
			return resize_ncols(ncols);
		}

		if (!_allocate(nrows, ncols)) {
			return false;
		}

		set_rows_value(0, nrows-1, v);
		return true;
	}

	inline void set_rows_value(index_row a, index_row b, const_reference v = value_type())
	{
		PVMatrixComputation<value_type, index_row, index_col>::set_rows_value(_data, a, b, _ncols, v);
	}

	void copy_to(matrix_type& dst) const
	{
		dst.reserve(_nrows, _ncols);
		if (value_pod::value) {
			memcpy(dst._data, _data, _ncols*_nrows*sizeof(value_type));
		}
		else {
			for (index_row i = 0; i < _nrows*_ncols; i++) {
				new (&dst._data[i]) value_type(_data[i]);
			}
		}

	}

	void swap(matrix_type& dst)
	{
		index_col ncols_tmp;
		index_row nrows_tmp;
		index_row nrows_physical_tmp;
		pointer data_tmp;

		ncols_tmp = _ncols;
		nrows_tmp = _nrows;
		nrows_physical_tmp = _nrows_physical;
		data_tmp = _data;

		_data = dst._data;
		_ncols = dst._ncols;
		_nrows = dst._nrows;
		_nrows_physical = dst._nrows_physical;

		dst._data = data_tmp;
		dst._ncols = ncols_tmp;
		dst._nrows = nrows_tmp;
		dst._nrows_physical = nrows_physical_tmp;
	}

	inline bool reserve(index_row nrows, index_col ncols)
	{
		return _allocate(nrows, ncols);
	}

	bool resize_nrows(index_row nrows)
	{
		assert(_data);
		if (nrows <= _nrows_physical) {
			// Just shrink the container
			_nrows = nrows;
		}
		else {
			// We need a reallocation
			if (!_reallocate_nrows(nrows)) {
				return false;
			}
		}
		return true;
	}

	bool resize_nrows(index_row nrows, const_reference v)
	{
		assert(_data);
		if (nrows <= _nrows) {
			// Just shrink the container
			_nrows = nrows;
		}
		else 
		if (nrows <= _nrows_physical) {
			// Reset the matrix' values
			set_rows_value(_nrows, nrows-1, v);
			_nrows = nrows;
		}
		else {
			// We need a reallocation
			index_row nrows_old = _nrows;
			if (!_reallocate_nrows(nrows)) {
				return false;
			}
			set_rows_value(nrows_old, nrows-1, v);
		}
		return true;
	}

	bool resize_ncols(index_col ncols)
	{
		index_col old_ncols = _ncols;
		if (!_reallocate_ncols(ncols)) {
			return false;
		}
		if (!value_pod::value && _ncols > old_ncols) {
			for (index_row i = 0; i < _nrows; i++) {
				for (index_col j = old_ncols; j < _ncols; j++) {
					new (&_data[i*_ncols+j]) value_type();
				}
			}
		}
		return true;
	}

	void free()
	{
		if (_data) {
			free_buf(_data);
			_data = NULL;
			_nrows = 0;
			_ncols = 0;
			_nrows_physical = 0;
		}
	}

public:

	pointer get_data() { return _data; }
	const_pointer get_data() const { return _data; }

	inline line get_row(index_row row) { assert(_data); assert(row < _nrows); return line(*this, row); }
	inline const_line get_row(index_row row) const { assert(_data); assert(row < _nrows); return const_line(*this, row); }

	inline line operator[](index_row row) { return get_row(row); }
	inline const_line operator[](index_row row) const { return get_row(row); }

	inline pointer get_row_ptr(index_row row) { assert(_data); assert(row < _nrows); return &_data[row*_ncols]; }
	inline const_pointer get_row_ptr(index_row row) const { assert(_data); assert(row < _nrows); return &_data[row*_ncols]; }

	inline reference at(index_row row, index_col col) { assert(row < _nrows); assert(col < _ncols); return _data[row*_ncols +col]; }
	inline const_reference at(index_row row, index_col col) const { assert(row < _nrows); assert(col < _ncols); return _data[row*_ncols +col]; }

	void set_value(index_row row, index_col col, const_reference v) { assert(row < _nrows); assert(col < _ncols); _data[row*_ncols + col] = v; }

	index_col get_width() const { return _ncols; }
	index_row get_height() const { return _nrows; }

	index_col get_ncols() const { return _ncols; }
	index_row get_nrows() const { return _nrows; }

public:
	void transpose_to(transposed_type& res)
	{
		res.resize(_ncols, _nrows);
		transpose_to(res._data);
	}

private:
	void transpose_to(pointer res)
	{
		PVMatrixComputation<value_type, index_row, index_col>::transpose(res, _data, _nrows, _ncols);
	}

	void free_buf(pointer p)
	{
		// Destruct objects
		if (!value_pod::value) {
#pragma omp parallel for
			for (int64_t i = 0; i < _nrows*_ncols; i++) {
				_alloc.destroy(&p[i]);
			}
		}
		_alloc.deallocate(p, _nrows*_ncols);
	}

	inline bool _allocate(index_row nrows, index_col ncols)
	{
		if (_data) {
			free_buf(_data);
		}
		pointer p;
		try {
			p = _alloc.allocate(nrows*ncols);
		}
		catch (std::bad_alloc const&) {
			PVLOG_ERROR("(PVMatrix::_allocate) unable to allocate %ld x %ld (%ld bytes).", nrows, ncols, nrows*ncols*sizeof(value_type));
			return false;
		}
		_data = p;
		if (p == NULL) {
			return false;
		}
		_nrows = nrows;
		_nrows_physical = nrows;
		_ncols = ncols;
		return true;
	}

	inline bool _reallocate_nrows(index_row nrows)
	{
		assert(_data);
		pointer p = _alloc.allocate(nrows*_ncols);
		if (p == NULL) {
			return false;
		}
		if (value_pod::value) {
			memcpy(p, _data, sizeof(value_type)*_nrows*_ncols);
		}
		else {
			for (index_row i = 0; i < _nrows*_ncols; i++) {
				new (&p[i]) value_type(_data[i]);
			}
		}
		free_buf(_data);
		_data = p;
		_nrows = nrows;
		_nrows_physical = nrows;
		return true;
	}

	inline bool _reallocate_ncols(index_col ncols)
	{
		assert(_data);

		if (ncols == _ncols) {
			return true;
		}

		pointer p = _alloc.allocate(_nrows*ncols);
		if (p == NULL) {
			return false;
		}
		index_col colmin = picviz_min(_ncols, ncols);
		if (value_pod::value) {
			for (index_row i = 0; i < _nrows; i++) {
				memcpy(&p[i*ncols], &_data[i*_ncols], colmin);
			}
		}
		else {
			for (index_row i = 0; i < _nrows; i++) {
				for (index_col j = 0; j < colmin; j++) {
					new (&p[i*ncols+j]) value_type(_data[i*_ncols+j]);
				}
			}
		}
		free_buf(_data);
		_data = p;
		_ncols = ncols;
		return true;
	}

private:
	index_col _ncols;
	index_row _nrows;
	index_row _nrows_physical;
	pointer _data;
	allocator_type _alloc;
};


}



#endif
