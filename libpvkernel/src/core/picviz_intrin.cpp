#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/sse4detector.h>

#include <assert.h>

static bool _has_sse41 = false;
static bool _has_sse42 = false;
static bool _init_done = false;

void init_cpuid()
{
	_has_sse41 = isSSE41Supported();
	_has_sse42 = isSSE41andSSE42supported();
	_init_done = true;
}


bool has_sse41()
{
	assert(_init_done);
	return _has_sse41;
}

bool has_sse42()
{
	assert(_init_done);
	return _has_sse42;
}
