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

#define STRINGIFY(X) #X

#define SHARED_MEMORY_SIZE (16 * 1024)
#define THREAD_NUM_COUNT 512

/******************************************************************************
 * opencl_kernel
 *****************************************************************************/

/* image is ARGB, not RGBA ;-)
 * The principle is the following:
 * - the shared memory contains a buffer
 *
 *
 * Inspector has its hue starting at blue while standard HSV model starts with red:
 * inspector: B C G Y R M B
 * standard : R Y G C B M R
 *
 * H_c = (N_color + R_i - H_i) mod N_color
 * where:
 * - H_c is the corrected hue value
 * - H_i is the hue value in Inspector
 * - N_color is the number of color (see HSV_COLOR_COUNT)
 * - R_i is the index of the red color in Inspector
 *
 * in hue2rgb(...), real computation of 'r' is:
 * -- code --
 * const float3 r = mix(K.xxx, clamp(p - K.xxx, value0, value1), c.y);
 * -- code --
 * but as c.y == 1.0, it can simplified
 *
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

	uint hn = (HSV_COLOR_COUNT + HSV_COLOR_RED - hue) % HSV_COLOR_COUNT;
	float4 c = (float4)(hn / (float)HSV_COLOR_COUNT, 1.0, 1.0, 1.0);

	const float3 value0 = (float)(0.0);
	const float3 value1 = (float)(1.0);

	const float4 K = (float4)(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);

	float3 dummy;
	const float3 p = fabs(fract(c.xxx + K.xyz, &dummy) * 6.0f - K.www);

	const float3 r = clamp(p - K.xxx, value0, value1);

	return 0xFF000000 | (uint)(0xFF * r.x) << 16 | (uint)(0xFF * r.y) << 8 | (uint)(0xFF * r.z);
}

kernel void draw_bci(const global uint2* bci_codes,
                     const uint n,
                     const uint width,
                     global uint* img_dst,
                     const uint img_width,
                     const uint image_height,
                     const uint img_x_start,
                     const float zoom_y,
                     const uint bit_shift,
                     const uint bit_mask,
                     const uint reverse)
{
	local uint shared_img[SHARED_MEMORY_SIZE / sizeof(uint)];

	int band_x = get_local_id(0) + get_group_id(0)*get_local_size(0);
	if (band_x >= width) {
		return;
	}

	const float alpha0 = (float)(width-band_x)/(float)width;
	const float alpha1 = (float)(width-(band_x+1))/(float)width;
	const uint y_start = get_local_id(1) + get_group_id(1)*get_local_size(1);
	const uint size_grid = get_local_size(1)*get_num_groups(1);

	for (int y = get_local_id(1); y < image_height; y += get_local_size(1)) {
		shared_img[get_local_id(0) + y*get_local_size(0)] = 0xFFFFFFFF;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int idx_codes = y_start;
	for (; idx_codes < n; idx_codes += size_grid) {
		uint2 code0 = bci_codes[idx_codes];
		code0.x &= 0xFFFFFF00;
		const float l0 = (float) (code0.y & bit_mask);
		const int r0i = (code0.y >> bit_shift) & bit_mask;
		const int type = (code0.y >> ((2*bit_shift) + 8)) & 3;
		int pixel_y00;
		int pixel_y01;
		if (type == 0) { // STRAIGHT
			const float r0 = (float) r0i;
			pixel_y00 = (int) (((r0 + ((l0-r0)*alpha0)) * zoom_y) + 0.5f);
			pixel_y01 = (int) (((r0 + ((l0-r0)*alpha1)) * zoom_y) + 0.5f);
		}
		else {
			if (band_x > r0i) {
				// This is out of our drawing scope!
				continue;
			}
			const float r0 = (float) r0i;
			if (type == 1) { // UP
				const float alpha_x = l0/r0;
				pixel_y00 = (int) (((l0-(alpha_x*(float)band_x))*zoom_y) + 0.5f);
				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				}
				else {
					pixel_y01 = (int) (((l0-(alpha_x*(float)(band_x+1)))*zoom_y) + 0.5f);
				}
			}
			else {
				const float alpha_x = ((float)bit_mask-l0)/r0;
				pixel_y00 = (int) (((l0+(alpha_x*(float)band_x))*zoom_y) + 0.5f);
				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				}
				else {
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

	band_x += img_x_start;
	if (reverse) {
		band_x = img_width-band_x-1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int y = get_local_id(1); y < image_height; y += get_local_size(1)) {
		const uint pixel_shared = shared_img[get_local_id(0) + y*get_local_size(0)];
		uint pixel;
		if (pixel_shared != 0xFFFFFFFF) {
			pixel = hue2rgb(pixel_shared & 0x000000FF);
		}
		else {
			pixel = 0x00000000;
		}
		img_dst[band_x + y*img_width] = pixel;
	}
}

                                      );
// clang-format on

template <size_t Bbits>
struct opencl_kernel {
	static cl_int start(const cl_command_queue queue,
	                    const cl_kernel kernel,
	                    const cl_mem bci_mem,
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
		const cl_uint real_reverse = reverse;
		const size_t column_mem_size = image_height * sizeof(cl_uint);

		inendi_verify_opencl(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bci_mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n));
		inendi_verify_opencl(clSetKernelArg(kernel, 2, sizeof(cl_uint), &width));
		inendi_verify_opencl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &image_mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 4, sizeof(cl_uint), &image_width));
		inendi_verify_opencl(clSetKernelArg(kernel, 5, sizeof(cl_uint), &image_height));
		inendi_verify_opencl(clSetKernelArg(kernel, 6, sizeof(cl_uint), &image_x_start));
		inendi_verify_opencl(clSetKernelArg(kernel, 7, sizeof(cl_float), &zoom_y));
		inendi_verify_opencl(clSetKernelArg(kernel, 8, sizeof(cl_uint), &bit_shift));
		inendi_verify_opencl(clSetKernelArg(kernel, 9, sizeof(cl_uint), &bit_mask));
		inendi_verify_opencl(clSetKernelArg(kernel, 10, sizeof(cl_uint), &real_reverse));

		/* we maximize usefull work group size horizontally
		 */
		const size_t local_num_x = std::min((size_t)width, SHARED_MEMORY_SIZE / column_mem_size);
		const size_t local_num_y = THREAD_NUM_COUNT / local_num_x;

		/* make a valid work items count (a multiple of work group size)
		 */
		const size_t global_num_x = ((width + local_num_x - 1) / local_num_x) * local_num_x;
		const size_t global_num_y = local_num_y;

		const size_t global_work[] = {global_num_x, global_num_y};
		const size_t local_work[] = {local_num_x, local_num_y};

#if 0
		std::cout << "########################################" << std::endl;
		std::cout << "global size : " << global_num_x << " " << global_num_y << std::endl;
		std::cout << "local size  : " << local_num_x << " " << local_num_y << std::endl;
		std::cout << "width       : " << width << std::endl;
		std::cout << "image_width : " << image_width << std::endl;
		std::cout << "image_height: " << image_height << std::endl;
		std::cout << "zoom_y      : " << zoom_y << std::endl;
		std::cout << "n           : " << n << std::endl;
#endif

		return clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work, local_work, 0,
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

	/* NOTE: 1 because the kernel code is stored in one null-terminated string.
	 */
	cl_program program = clCreateProgramWithSource(_context, 1, &source, nullptr, &err);
	inendi_verify_opencl_var(err);

	/**
	 * NOTE: options can be passed to build process, like -DVAR=VAL. So that, Bbits
	 * dependant values and reverse can be passed at build time to decrease parameter
	 * count and have better optimisations.
	 */
	std::stringstream build_options;
	build_options << "-DSHARED_MEMORY_SIZE=" << SHARED_MEMORY_SIZE;
	build_options << " -DHSV_COLOR_COUNT=" << HSV_COLOR_COUNT;
	build_options << " -DHSV_COLOR_WHITE=" << HSV_COLOR_WHITE;
	build_options << " -DHSV_COLOR_BLACK=" << HSV_COLOR_BLACK;
	build_options << " -DHSV_COLOR_RED=" << HSV_COLOR_RED;

	err = clBuildProgram(program, 0, nullptr, build_options.str().c_str(), nullptr, nullptr);

	_kernel = clCreateKernel(program, "draw_bci", &err);
	inendi_verify_opencl_var(err);

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
PVParallelView::PVBCIDrawingBackendOpenCL::create_image(size_t img_width, uint8_t height_bits)
{
	assert(_devices.size() >= 1);

	if (_next_device == _devices.end()) {
		_next_device = _devices.begin();
	}

	// Create image on a device in a round robin way
	cl_command_queue queue = _next_device->second.queue;

	PVBCIBackendImage_p ret(
	    new PVBCIBackendImageOpenCL(img_width, height_bits, _context, queue, _next_device->first));

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

		err = opencl_kernel<10>::start(dev.queue, _kernel, dev.mem, n, width, dst_img->device_mem(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	case 11:
		err = opencl_kernel<11>::start(dev.queue, _kernel, dev.mem, n, width, dst_img->device_mem(),
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

	BENCH_END(ocl_kernel, "OCL kernel", 1, 1, 1, 1);
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
