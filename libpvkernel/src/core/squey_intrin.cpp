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

#include <pvkernel/core/squey_intrin.h>
#include <pvkernel/core/sse4detector.h>

#include <cassert>

bool PVCore::PVIntrinsics::_has_sse41 = false;
bool PVCore::PVIntrinsics::_has_sse42 = false;
bool PVCore::PVIntrinsics::_init_done = false;

void PVCore::PVIntrinsics::init_cpuid()
{
	_has_sse41 = isSSE41Supported();
	_has_sse42 = isSSE41andSSE42supported();
	_init_done = true;
}

bool PVCore::PVIntrinsics::has_sse41()
{
	assert(_init_done);
	return _has_sse41;
}

bool PVCore::PVIntrinsics::has_sse42()
{
	assert(_init_done);
	return _has_sse42;
}
