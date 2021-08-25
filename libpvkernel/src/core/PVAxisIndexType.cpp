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

#include <pvkernel/core/PVAxisIndexType.h>

PVCore::PVAxisIndexType::PVAxisIndexType()
{
	_origin_axis_index = PVCol(-1);
	_append_none_axis = false;
}

PVCore::PVAxisIndexType::PVAxisIndexType(PVCol origin_axis_index,
                                         bool append_none_axis,
                                         PVCombCol axis_index /* = PVCombCol(0) */)
{
	_origin_axis_index = origin_axis_index;
	_append_none_axis = append_none_axis;
	_axis_index = axis_index;
}

PVCol PVCore::PVAxisIndexType::get_original_index()
{
	return _origin_axis_index;
}

bool PVCore::PVAxisIndexType::get_append_none_axis()
{
	return _append_none_axis;
}

PVCombCol PVCore::PVAxisIndexType::get_axis_index()
{
	return _axis_index;
}
