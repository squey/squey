#include <pvkernel/core/PVBufferSlice.h>
#include <cassert>

#include <tbb/scalable_allocator.h>

#include <pvkernel/core/stdint.h>

#define REALLOC_GROWBY_ADD 20

// No buffer simulation
static uint32_t g_null_buf_data = 0;
static char* g_null_buf = (char*) &g_null_buf_data;
static char* g_null_buf_end = ((char*) &g_null_buf_data)+sizeof(uint32_t);


PVCore::PVBufferSlice::PVBufferSlice(buf_list_t& buf_list) :
	_buf_list(buf_list)
{
	_begin = g_null_buf;
	_end = g_null_buf_end;
	_physical_end = g_null_buf_end;
	_realloc_buf = NULL;
}

PVCore::PVBufferSlice::PVBufferSlice(char* begin, char* end, buf_list_t& buf_list) :
	_buf_list(buf_list)
{
	assert(end >= begin);
	_begin = begin;
	_end = end;
	_physical_end = end;
	_realloc_buf = NULL;
}

void PVCore::PVBufferSlice::set_begin(char* p)
{
	_begin = p;
}

void PVCore::PVBufferSlice::set_end(char* p)
{
	assert(p >= _begin);
	_end = p;
}

void PVCore::PVBufferSlice::set_physical_end(char* p)
{
	assert(p >= _end);
	_physical_end = p;
}

bool PVCore::PVBufferSlice::grow_by(size_t n)
{
	if (n == 0) {
		n = (uintptr_t) _physical_end - (uintptr_t) _end;
	}
	else
	if (_end + n > _physical_end) {
		return false;
	}
	set_end(_end + n);
	return true;
}

void PVCore::PVBufferSlice::grow_by_reallocate(size_t n)
{
	if (grow_by(n))
		return;
	// No choice here...
	// TODO: allocate this in a circular buffer
	
	static tbb::scalable_allocator<char> alloc;
	size_t s = size();
	char* new_buf;
	if (_realloc_buf) {
		_buf_list.remove(buf_list_t::value_type(_realloc_buf, _physical_end - _begin));
		char* tmp_new = alloc.allocate(s + n + REALLOC_GROWBY_ADD);
		_buf_list.push_back(buf_list_t::value_type(tmp_new, s+n+REALLOC_GROWBY_ADD));
		memcpy(tmp_new, _begin, s);
		new_buf = tmp_new;
		alloc.deallocate(_realloc_buf, _physical_end - _begin);
		_realloc_buf = tmp_new;
	}
	else {
		_realloc_buf = alloc.allocate(s + n + REALLOC_GROWBY_ADD);
		_buf_list.push_back(buf_list_t::value_type(_realloc_buf, s+n+REALLOC_GROWBY_ADD));
		new_buf = _realloc_buf;
		memcpy(new_buf, _begin, s);
	}
	memset(new_buf+s, 0, n);
	_begin = new_buf;
	_end = new_buf+s+n;
	_physical_end = _end + REALLOC_GROWBY_ADD;
	//init_qstr();
}

void PVCore::PVBufferSlice::allocate_new(size_t n)
{
	// Reallocate new buffer with size 'n', and discards the old content
	static tbb::scalable_allocator<char> alloc;
	if (_realloc_buf) {
		_buf_list.remove(buf_list_t::value_type(_realloc_buf, _physical_end - _begin));
	}
	_begin = alloc.allocate(n);
	_end = _begin+n;
	_physical_end = _end;
	_buf_list.push_back(buf_list_t::value_type(_begin, n));
}

void PVCore::PVBufferSlice::_realloc_data()
{
	static tbb::scalable_allocator<char> alloc;
	size_t s = (uintptr_t)_physical_end - (uintptr_t)_begin;
	size_t old_size = size();

	_realloc_buf = alloc.allocate(s);
	_buf_list.push_back(buf_list_t::value_type(_realloc_buf, s));

	char* new_buf;
	new_buf = _realloc_buf;
	memcpy(new_buf, _begin, s);

	_begin = new_buf;
	_end = new_buf+old_size;
	_physical_end = new_buf+s;
}

size_t PVCore::PVBufferSlice::size() const
{
	return (size_t) ((uintptr_t)_end - (uintptr_t)_begin);
}

char* PVCore::PVBufferSlice::begin() const
{
	return _begin;
}

char* PVCore::PVBufferSlice::end() const
{
	return _end;
}
