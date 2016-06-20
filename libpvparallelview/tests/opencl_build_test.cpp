
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>

/**
 * This test aims to check that the kernel builds without error after the
 * obfuscation pass.
 */
int main()
{
	using namespace PVParallelView;

	PVParallelViewImpl::get().init_backends<PVBCIDrawingBackendOpenCL>();

	return 0;
}
