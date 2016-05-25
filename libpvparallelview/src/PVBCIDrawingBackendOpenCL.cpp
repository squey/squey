/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVConfig.h>

#include <pvkernel/opencl/common.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIBackendImageOpenCL.h>

#include <pvparallelview/PVBCICode.h>

#include <cassert>

#include <iostream>
#include <sstream>

/* a simple macro to embed code as a nul-terminated string
 */
#define STRINGIFY(X) #X

/* minimal required for OpenCL 1.0 devices
 */
#define LOCAL_MEMORY_SIZE (16 * 1024)

/******************************************************************************
 * opencl_kernel
 *****************************************************************************/

/* As the kernel code is embedded in the present C++ file, the comments are
 * not inline.
 *
 * Principle
 *
 * To minimize global memory accesses, thre kernel uses computing unit local memory area to do the
 * collisions in a first step and copy data from the local memory to the global memory are in a
 * second one. A preliminary step is local memory initialization.
 *
 * Final image is processed column by column (don't know why).
 *
 * To reduce local memory initialization/copy, more than one image column is processed at the same
 * time. This number of image column depend of the column number which can fit in the local memory
 * area (for 1024 pixels height image, it's 4).
 *
 * For each image column, the kernel will process BCI codes in parallel to find the corresponding
 * pixel in the column and do the collisions on their index value.
 *
 * When the final image is taller than broad, a BCI code can lead to more than one pixel in an image
 * column.
 *
 * Due to zoomed pararallel coordinate view, there are 3 types of BCI codes:
 * - "straight" ones (whose type is 0) which hit the final image right border;
 * - "up" ones (whose type is 1) which hit the final image top border;
 * - "down" one which hit the final image top border.
 *
 * Notes:
 *
 * QImage are ARGB, not RGBA ;-)
 *
 * Inspector has its hue starting at blue while standard HSV model starts with red:
 * inspector: B C G Y R M B
 * standard : R Y G C B M R
 *
 * H_c = (N_color + R_i - H_i) mod N_color
 * where:
 * - H_c is the correct hue value
 * - H_i is the Inspector hue value
 * - N_color is the number of color (see HSV_COLOR_COUNT)
 * - R_i is the index of the red color in Inspector
 *
 * in hue2rgb(...), real computation of 'r' is:
 * -- code --
 * const float3 r = mix(K.xxx, clamp(p - K.xxx, value0, value1), c.y);
 * -- code --
 * but as c.y is always equal to 1.0 in our case, the expression can be simplified into
 * -- code --
 * const float3 r = clamp(p - K.xxx, value0, value1);
 * -- code --
 */

// clang-format off
static const char* source = STRINGIFY(

uint hue2rgb(uint hue)
{
	if (hue == HSV_COLOR_WHITE) {
		return 0xFFFFFFFF;
	}
	if (hue == HSV_COLOR_BLACK) {
		return 0xFF000000;
	}

	uint nh = (HSV_COLOR_COUNT + HSV_COLOR_RED - hue) % HSV_COLOR_COUNT;
	float4 c = (float4)(nh / (float)HSV_COLOR_COUNT, 1.0, 1.0, 1.0);

	const float3 value0 = (float)(0.0);
	const float3 value1 = (float)(1.0);

	const float4 K = (float4)(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);

	float3 dummy;
	const float3 p = fabs(fract(c.xxx + K.xyz, &dummy) * 6.0f - K.www);

	const float3 r = clamp(p - K.xxx, value0, value1);

	return 0xFF000000 | (uint)(0xFF * r.x) << 16 | (uint)(0xFF * r.y) << 8 | (uint)(0xFF * r.z);
}

kernel void draw(const global uint2* bci_codes,
                 const uint n,
                 const uint width,
                 global uint* image,
                 const uint image_width,
                 const uint image_height,
                 const uint image_x_start,
                 const float zoom_y,
                 const uint bit_shift,
                 const uint bit_mask,
                 const uint reverse)
{
	local uint shared_img[LOCAL_MEMORY_SIZE / sizeof(uint)];

	int band_x = get_local_id(0) + get_group_id(0)*get_local_size(0);

	if (band_x >= width) {
		return;
	}

	const float alpha0 = (float)(width-band_x)/(float)width;
	const float alpha1 = (float)(width-(band_x+1))/(float)width;
	const uint y_start = get_local_id(1) + get_group_id(1)*get_local_size(1);
	const uint y_pitch = get_local_size(1)*get_num_groups(1);

	for (int y = get_local_id(1); y < image_height; y += get_local_size(1)) {
		shared_img[get_local_id(0) + y*get_local_size(0)] = 0xFFFFFFFF;
	}

	int pixel_y00;
	int pixel_y01;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (uint idx_codes = y_start; idx_codes < n; idx_codes += y_pitch) {
		uint2 code0 = bci_codes[idx_codes];

		code0.x &= 0xFFFFFF00;

		const float l0 = (float) (code0.y & bit_mask);
		const int r0i = (code0.y >> bit_shift) & bit_mask;
		const int type = (code0.y >> ((2*bit_shift) + 8)) & 3;

		if (type == 0) { // STRAIGHT
			const float r0 = (float) r0i;

			pixel_y00 = (int) (((r0 + ((l0-r0)*alpha0)) * zoom_y) + 0.5f);
			pixel_y01 = (int) (((r0 + ((l0-r0)*alpha1)) * zoom_y) + 0.5f);
		} else {
			if (band_x > r0i) {
				continue;
			}

			const float r0 = (float) r0i;

			if (type == 1) { // UP
				const float alpha_x = l0/r0;

				pixel_y00 = (int) (((l0-(alpha_x*(float)band_x))*zoom_y) + 0.5f);

				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				} else {
					pixel_y01 = (int) (((l0-(alpha_x*(float)(band_x+1)))*zoom_y) + 0.5f);
				}
			} else {
				const float alpha_x = ((float)bit_mask-l0)/r0;

				pixel_y00 = (int) (((l0+(alpha_x*(float)band_x))*zoom_y) + 0.5f);

				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				} else {
					pixel_y01 = (int) (((l0+(alpha_x*(float)(band_x+1)))*zoom_y) + 0.5f);
				}
			}
		}

		pixel_y00 = clamp(pixel_y00, 0, (int)image_height);
		pixel_y01 = clamp(pixel_y01, 0, (int)image_height);

		if (pixel_y00 > pixel_y01) {
			const int tmp = pixel_y00;

			pixel_y00 = pixel_y01;
			pixel_y01 = tmp;
		}

		const uint color0 = (code0.y >> 2*bit_shift) & 0xFF;

		if (color0 == HSV_COLOR_BLACK) {
			code0.x = 0xFFFFFF00;
		}

		const uint shared_v = color0 | code0.x;

		atomic_min(&shared_img[get_local_id(0) + pixel_y00*get_local_size(0)], shared_v);

		for (int pixel_y0 = pixel_y00+1; pixel_y0 < pixel_y01; pixel_y0++) {
			atomic_min(&shared_img[get_local_id(0) + pixel_y0*get_local_size(0)], shared_v);
		}
	}

	band_x += image_x_start;

	if (reverse) {
		band_x = image_width-band_x-1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int y = get_local_id(1); y < image_height; y += get_local_size(1)) {
		const uint pixel_shared = shared_img[get_local_id(0) + y*get_local_size(0)];
		uint pixel;

		if (pixel_shared != 0xFFFFFFFF) {
			pixel = hue2rgb(pixel_shared & 0x000000FF);
		} else {
			pixel = 0x00000000;
		}
		image[band_x + y*image_width] = pixel;
	}
}

                                      );
// clang-format on

template <size_t Bbits>
struct opencl_kernel {
	static cl_int start(const PVParallelView::PVBCIDrawingBackendOpenCL::device_t& dev,
	                    const cl_kernel kernel,
	                    const cl_uint n,
	                    const cl_uint width,
	                    const cl_mem image_mem,
	                    const cl_uint image_width,
	                    const cl_uint image_x_start,
	                    const cl_float zoom_y,
	                    const bool reverse)
	{
		const cl_uint bit_shift = Bbits;
		const cl_uint bit_mask = PVParallelView::constants<Bbits>::mask_int_ycoord;
		const cl_uint image_height = PVParallelView::constants<Bbits>::image_height;
		const size_t column_mem_size = image_height * sizeof(cl_uint);
		// bool is not a valid type as kernel parameter
		const cl_uint reverse_flag = reverse;

		inendi_verify_opencl(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev.mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n));
		inendi_verify_opencl(clSetKernelArg(kernel, 2, sizeof(cl_uint), &width));
		inendi_verify_opencl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &image_mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 4, sizeof(cl_uint), &image_width));
		inendi_verify_opencl(clSetKernelArg(kernel, 5, sizeof(cl_uint), &image_height));
		inendi_verify_opencl(clSetKernelArg(kernel, 6, sizeof(cl_uint), &image_x_start));
		inendi_verify_opencl(clSetKernelArg(kernel, 7, sizeof(cl_float), &zoom_y));
		inendi_verify_opencl(clSetKernelArg(kernel, 8, sizeof(cl_uint), &bit_shift));
		inendi_verify_opencl(clSetKernelArg(kernel, 9, sizeof(cl_uint), &bit_mask));
		inendi_verify_opencl(clSetKernelArg(kernel, 10, sizeof(cl_uint), &reverse_flag));

		/* we make fit the highest number of image column in the work group local memory
		 */
		const size_t local_num_x = std::min((size_t)width, LOCAL_MEMORY_SIZE / column_mem_size);
		const size_t local_num_y = dev.work_group_size / local_num_x;

		const size_t global_num_x = ((width + local_num_x - 1) / local_num_x) * local_num_x;
		const size_t global_num_y = local_num_y;

		const size_t global_work[] = {global_num_x, global_num_y};
		const size_t local_work[] = {local_num_x, local_num_y};

		return clEnqueueNDRangeKernel(dev.queue, kernel, 2, nullptr, global_work, local_work, 0,
		                              nullptr, nullptr);
	}
};

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::PVBCIDrawingBackendOpenCL
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL::PVBCIDrawingBackendOpenCL()
    : _context(nullptr), _is_software(false)
{
	size_t size = PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase);
	int dev_idx = 0;
	cl_int err;

	auto& config = PVCore::PVConfig::get().config();
	bool force_cpu = config.value("backend_opencl/force_cpu", false).toBool();

	// List all usable OpenCL devices and create appropriate structures
	const auto fun = [&](cl_context ctx, cl_device_id dev_id) {
		device_t dev;
		cl_int err;

		dev.id = dev_id;

		dev.queue = clCreateCommandQueue(ctx, dev_id, 0, &err);
		inendi_verify_opencl_var(err);

		dev.addr = PVOpenCL::host_allocate<PVBCICodeBase>(ctx, dev.queue, CL_MEM_READ_ONLY,
		                                                  CL_MAP_WRITE, size, dev.mem, err);
		inendi_verify_opencl_var(err);
		assert(dev.addr != nullptr);

		this->_devices.insert(std::make_pair(dev_idx, dev));
		++dev_idx;
	};

	if (force_cpu == false) {
		_context = PVOpenCL::find_first_usable_context(true, fun);
	}

	if (_context == nullptr) {
		_context = PVOpenCL::find_first_usable_context(false, fun);
		_is_software = true;
	}

	if (_context == nullptr) {
		throw PVOpenCL::exception::no_backend_error();
	}

	_next_device = _devices.begin();

	/* NOTE: "1" because the kernel code is stored using an array of nul-terminated string.
	 */
	cl_program program = clCreateProgramWithSource(_context, 1, &source, nullptr, &err);
	inendi_verify_opencl_var(err);

	/**
	 * NOTE: options can be passed to build process, like -DVAR=VAL. So that, Bbits
	 * dependant values and reverse can be passed at build time to decrease parameter
	 * count and have better optimisations.
	 */
	std::stringstream build_options;
	build_options << "-DLOCAL_MEMORY_SIZE=" << LOCAL_MEMORY_SIZE;
	build_options << " -DHSV_COLOR_COUNT=" << HSV_COLOR_COUNT;
	build_options << " -DHSV_COLOR_WHITE=" << HSV_COLOR_WHITE;
	build_options << " -DHSV_COLOR_BLACK=" << HSV_COLOR_BLACK;
	build_options << " -DHSV_COLOR_RED=" << HSV_COLOR_RED;

	err = clBuildProgram(program, 0, nullptr, build_options.str().c_str(), nullptr, nullptr);

#ifdef INENDI_DEVELOPER_MODE
	if (err != CL_SUCCESS) {
		/* As we build (implicitly) on all devices, we check every for errors
		 */
		for (auto& it : _devices) {
			cl_build_status status;

			clGetProgramBuildInfo(program, it.second.id, CL_PROGRAM_BUILD_STATUS, sizeof(status),
			                      &status, nullptr);

			if (status != CL_BUILD_ERROR) {
				continue;
			}

			size_t len;
			char buffer[256];
			clGetProgramBuildInfo(program, it.second.id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
			                      buffer, &len);
			PVLOG_INFO("build log: %s\n", buffer);
		}
	}
#endif
	inendi_verify_opencl_var(err);

	_kernel = clCreateKernel(program, "draw", &err);
	inendi_verify_opencl_var(err);

	for (auto& it : _devices) {
		err = clGetKernelWorkGroupInfo(_kernel, it.second.id, CL_KERNEL_WORK_GROUP_SIZE,
		                               sizeof(size_t), &it.second.work_group_size, nullptr);
		inendi_verify_opencl_var(err);
	}

	err = clReleaseProgram(program);
	inendi_verify_opencl_var(err);
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::~PVBCIDrawingBackendOpenCL
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL::~PVBCIDrawingBackendOpenCL()
{
	for (auto& dev : _devices) {
		inendi_verify_opencl_var(
		    PVOpenCL::host_free(dev.second.queue, dev.second.mem, dev.second.addr));
		inendi_verify_opencl(clReleaseCommandQueue(dev.second.queue));
	}

	clReleaseContext(_context);
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::get
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL& PVParallelView::PVBCIDrawingBackendOpenCL::get()
{
	static PVBCIDrawingBackendOpenCL backend;
	return backend;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::create_image
 *****************************************************************************/

PVParallelView::PVBCIBackendImage_p
PVParallelView::PVBCIDrawingBackendOpenCL::create_image(size_t image_width, uint8_t height_bits)
{
	assert(_devices.size() >= 1);

	if (_next_device == _devices.end()) {
		_next_device = _devices.begin();
	}

	// Create image on a device in a round robin way
	cl_command_queue queue = _next_device->second.queue;

	PVBCIBackendImage_p ret(new PVBCIBackendImageOpenCL(image_width, height_bits, _context, queue,
	                                                    _next_device->first));

	++_next_device;

	return ret;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::allocate_bci
 *****************************************************************************/

PVParallelView::PVBCICodeBase* PVParallelView::PVBCIDrawingBackendOpenCL::allocate_bci(size_t n)
{
	const size_t size = n * sizeof(PVParallelView::PVBCICodeBase);

	device_t dev;
	cl_int err;

	dev.queue = _devices[0].queue;
	dev.addr = PVOpenCL::host_allocate<PVBCICodeBase>(_context, dev.queue, CL_MEM_READ_WRITE,
	                                                  CL_MAP_READ, size, dev.mem, err);
	inendi_verify_opencl_var(err);
	assert(dev.addr != nullptr);

	_mapped_buffers.insert(std::make_pair(dev.addr, dev));

	return dev.addr;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::free_bci
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::free_bci(PVParallelView::PVBCICodeBase* buf)
{
	auto entry = _mapped_buffers.find(buf);

	if (entry == _mapped_buffers.end()) {
		assert(false);
		return;
	}

	cl_int err = PVOpenCL::host_free(entry->second.queue, entry->second.mem, entry->second.addr);
	inendi_verify_opencl_var(err);

	_mapped_buffers.erase(buf);
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::operator()
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::operator()(PVBCIBackendImage_p& backend_img,
                                                           size_t x_start,
                                                           size_t width,
                                                           PVBCICodeBase* codes,
                                                           size_t n,
                                                           const float zoom_y,
                                                           bool reverse,
                                                           std::function<void()> const& render_done)
{
#ifdef NDEBUG
	backend_image_t* dst_img = static_cast<backend_image_t*>(backend_img.get());
#else
	backend_image_t* dst_img = dynamic_cast<backend_image_t*>(backend_img.get());
	assert(dst_img != NULL);
#endif
	device_t& dev = _devices[dst_img->index()];

	cl_int err;

	BENCH_START(ocl_kernel);

	if (n != 0) {
		// Specs that a size of zero will lead to CL_INVALID_VALUE
		err = clEnqueueWriteBuffer(dev.queue, dev.mem, CL_FALSE, 0, n * sizeof(codes), codes, 0,
		                           nullptr, nullptr);
		inendi_verify_opencl_var(err);
	}

	switch (dst_img->height_bits()) {
	case 10:
		assert(reverse == false && "no reverse mode allowed in kernel<10>");

		err = opencl_kernel<10>::start(dev, _kernel, n, width, dst_img->device_mem(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	case 11:
		err = opencl_kernel<11>::start(dev, _kernel, n, width, dst_img->device_mem(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	default:
		assert(false);
		break;
	};
	inendi_verify_opencl_var(err);

	opencl_job_data_t* data = new opencl_job_data_t;
	data->done_function = render_done;

	dst_img->copy_device_to_host_async(&data->event);

	err = clSetEventCallback(data->event, CL_COMPLETE, &PVBCIDrawingBackendOpenCL::termination_cb,
	                         data);
	inendi_verify_opencl_var(err);

	// CPU drivers need to do an explicit clFlush to make event happen... strange...
	if (_is_software) {
		clFlush(dev.queue);
	}

	BENCH_END(ocl_kernel, "OCL kernel", n, sizeof(PVBCICodeBase), 1, 1);
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::wait_all
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::wait_all() const
{
	// Wait for all devices processing termination
	for (auto& device : _devices) {
		clFinish(device.second.queue);
	}
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::termination_cb
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::termination_cb(cl_event /* event */,
                                                               cl_int /* status */,
                                                               void* data)
{
	opencl_job_data_t* job_data = reinterpret_cast<opencl_job_data_t*>(data);

	// Call termination function
	if (job_data->done_function) {
		(job_data->done_function)();
	}

	delete job_data;
}
