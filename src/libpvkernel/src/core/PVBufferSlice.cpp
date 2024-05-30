//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVBufferSlice.h>

#include <tbb/tbb_allocator.h>

#include <cassert>
#include <cstring>

constexpr size_t REALLOC_GROWBY_ADD = 20;

// No buffer simulation
static uint32_t g_null_buf_data = 0;
static char* g_null_buf = (char*)&g_null_buf_data;
static char* g_null_buf_end = ((char*)&g_null_buf_data) + sizeof(uint32_t);

PVCore::PVBufferSlice::PVBufferSlice(buf_list_t& buf_list) : _buf_list(buf_list)
{
	_begin = g_null_buf;
	_end = g_null_buf_end;
	_physical_end = g_null_buf_end;
	_realloc_buf = nullptr;
}

PVCore::PVBufferSlice::PVBufferSlice(char* begin, char* end, buf_list_t& buf_list)
    : _buf_list(buf_list)
{
	if (end >= begin) {
		_begin = begin;
		_end = end;
		_physical_end = end;
	} else {
		_begin = g_null_buf;
		_end = g_null_buf_end;
		_physical_end = g_null_buf_end;
	}
	_realloc_buf = nullptr;
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
		n = (uintptr_t)_physical_end - (uintptr_t)_end;
	} else if (_end + n > _physical_end) {
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

	static tbb::tbb_allocator<char> alloc;
	size_t s = size();
	char* new_buf;
	if (_realloc_buf) {
		_buf_list.remove(buf_list_t::value_type(_realloc_buf, _physical_end - _begin));
		char* tmp_new = alloc.allocate(s + n + REALLOC_GROWBY_ADD);
		_buf_list.push_back(buf_list_t::value_type(tmp_new, s + n + REALLOC_GROWBY_ADD));
		memcpy(tmp_new, _begin, s);
		new_buf = tmp_new;
		alloc.deallocate(_realloc_buf, _physical_end - _begin);
		_realloc_buf = tmp_new;
	} else {
		_realloc_buf = alloc.allocate(s + n + REALLOC_GROWBY_ADD);
		_buf_list.push_back(buf_list_t::value_type(_realloc_buf, s + n + REALLOC_GROWBY_ADD));
		new_buf = _realloc_buf;
		memcpy(new_buf, _begin, s);
	}
	memset(new_buf + s, 0, n);
	_begin = new_buf;
	_end = new_buf + s + n;
	_physical_end = _end + REALLOC_GROWBY_ADD;
	// init_qstr();
}

void PVCore::PVBufferSlice::allocate_new(size_t n)
{
	// Reallocate new buffer with size 'n', and discards the old content
	static tbb::tbb_allocator<char> alloc;
	if (_realloc_buf) {
		_buf_list.remove(buf_list_t::value_type(_realloc_buf, _physical_end - _begin));
	}
	_begin = alloc.allocate(n);
	_end = _begin + n;
	_physical_end = _end;
	_buf_list.push_back(buf_list_t::value_type(_begin, n));
}

size_t PVCore::PVBufferSlice::size() const
{
	return (size_t)((uintptr_t)_end - (uintptr_t)_begin);
}

char* PVCore::PVBufferSlice::begin() const
{
	return _begin;
}

char* PVCore::PVBufferSlice::end() const
{
	return _end;
}
