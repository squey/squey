
#include <pvkernel/opencl/common.h>

#include <string>
#include <iostream>

static void print_device_info(cl_context ctx, cl_device_id dev_id)
{
	(void)ctx;
	size_t vsize;

	std::string str;

	clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, nullptr, &vsize);
	str.resize(vsize);
	clGetDeviceInfo(dev_id, CL_DEVICE_NAME, vsize, &str[0], nullptr);

	std::cout << "  device name: " << str << std::endl;
}

int main()
{
	cl_context ctx;

	std::cout << "testing with CPU device(s):" << std::endl;
	ctx = PVOpenCL::find_first_usable_context(CL_DEVICE_TYPE_CPU, print_device_info);
	clReleaseContext(ctx);

	std::cout << "testing GPU device(s):" << std::endl;
	ctx = PVOpenCL::find_first_usable_context(CL_DEVICE_TYPE_GPU, print_device_info);
	clReleaseContext(ctx);

	return 0;
}
