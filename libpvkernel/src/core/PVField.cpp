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

#include <pvkernel/core/PVBufferSlice.h> // for PVBufferSlice
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

PVCore::PVField::PVField(PVCore::PVElement& parent) : PVBufferSlice(parent.realloc_bufs())
{
	init(parent);
}

PVCore::PVField::PVField(PVCore::PVElement& parent, char* begin, char* end)
    : PVBufferSlice(begin, end, parent.realloc_bufs())
{
	init(parent);
}

void PVCore::PVField::init(PVElement& parent)
{
	_valid = true;
	_filtered = false;
	_parent = &parent;
}

bool PVCore::PVField::valid() const
{
	return _valid;
}

void PVCore::PVField::set_invalid()
{
	_valid = false;
}

bool PVCore::PVField::filtered() const
{
	return _filtered;
}

void PVCore::PVField::set_filtered()
{
	_filtered = true;
}

PVCore::PVElement* PVCore::PVField::elt_parent()
{
	return _parent;
}
