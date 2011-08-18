#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/sse4detector.h>

#include <assert.h>

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
