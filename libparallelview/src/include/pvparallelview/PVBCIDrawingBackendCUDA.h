/**
 * \file PVBCIDrawingBackendCUDA.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H

#include <pvkernel/cuda/common.h>

#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVBCIDrawingBackendSerial.h>
#include <pvparallelview/cuda/bci_cuda.h>

#include <npp.h> // NVIDIA's NPP library, for image resizing

#include <QImage>
#include <cassert>

#include <boost/thread.hpp>
#include <iostream>

namespace boost {
class thread;
}

namespace PVParallelView {

template <size_t Bbits>
class PVBCIDrawingBackendCUDA;

namespace __impl {

class cuda_engine;

template <size_t Bbits = NBITS_INDEX>
class PVBCIBackendImageCUDA: public PVParallelView::PVBCIBackendImage<Bbits>
{
	friend class PVParallelView::PVBCIDrawingBackendCUDA<Bbits>;
	friend class cuda_engine;
	
	typedef uint32_t pixel_t;
	typedef uint32_t* pixel_pointer_t;
	typedef uint32_t const* const_pixel_pointer_t;

	typedef std::allocator<pixel_t> pixel_allocator;

protected:
	PVBCIBackendImageCUDA(const uint32_t width, const int cuda_device):
		PVBCIBackendImage<Bbits>(width),
		_cuda_device(cuda_device)
	{
		size_t simg = PVBCIBackendImage<Bbits>::size_pixel();
		_host_img = pixel_allocator().allocate(simg);
		set_current_device();
		picviz_verify_cuda(cudaMalloc(&_device_img, simg*sizeof(pixel_t)));
		picviz_verify_cuda(cudaMemset(_device_img, 0, simg*sizeof(pixel_t)));
		_org_width = width;
	}

public:
	virtual ~PVBCIBackendImageCUDA()
	{
		size_t simg = PVBCIBackendImage<Bbits>::size_pixel();
		pixel_allocator().deallocate(_host_img, simg);
		picviz_verify_cuda(cudaFree(_device_img));
	}

	virtual void resize_width(PVBCIBackendImage<Bbits>& dst, const uint32_t width) const override
	{
		assert(org_width() % 4 == 0);

		PVBCIBackendImageCUDA<Bbits>* dst_img = static_cast<PVBCIBackendImageCUDA<Bbits>*>(&dst);
		assert(dst_img->org_width() % 4 == 0);

		set_current_device();

		cudaEvent_t end;
		cudaEventCreate(&end);

		// NPP resize method
		NppiRect rorg;
		rorg.x = 0; rorg.y = 0;
		rorg.width = PVBCIBackendImage<Bbits>::width(); rorg.height = PVBCIBackendImage<Bbits>::height();
		NppiSize sorg, sdst;
		sorg.width = PVBCIBackendImage<Bbits>::width(); sorg.height = PVBCIBackendImage<Bbits>::height();
		sdst.width = width; sdst.height = sorg.height;
		nppiResize_8u_C4R((const Npp8u*) device_img(), sorg, org_width()*sizeof(pixel_t), rorg, (Npp8u*) dst_img->device_img(), dst_img->org_width()*sizeof(pixel_t), sdst, (double)width/(double)sorg.width, 1.0, NPPI_INTER_NN);

		// wait for the end
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventDestroy(end);

		dst_img->set_width(width);
		dst_img->copy_device_to_host();
	}

protected:
	pixel_pointer_t device_img() { return _device_img; }
	pixel_pointer_t host_img() { return _host_img; }

	const_pixel_pointer_t device_img() const { return _device_img; }
	const_pixel_pointer_t host_img() const { return _host_img; }

	inline void copy_device_to_host(cudaStream_t const& stream) const
	{
		picviz_verify_cuda(cudaMemcpyAsync(_host_img, _device_img, size_org_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost, stream));
	}

	inline void copy_device_to_host() const
	{
		picviz_verify_cuda(cudaMemcpy(_host_img, _device_img, size_org_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost));
	}

	inline size_t org_width() const { return _org_width; }

	void set_current_device() const
	{
		picviz_verify_cuda(cudaSetDevice(_cuda_device));
	}

	int get_cuda_device() const { return _cuda_device; }

public:
	virtual QImage qimage() const
	{
		// This is really important to use the const constructor version of QImage, so that no data copy is made !
		return QImage((const uchar*) _host_img, PVBCIBackendImage<Bbits>::width(), PVBCIBackendImage<Bbits>::height(), _org_width*sizeof(uint32_t), QImage::Format_ARGB32_Premultiplied);
	}

	virtual bool set_width(uint32_t width)
	{
		if (width > _org_width) {
			return false;
		}

		return PVBCIBackendImage<Bbits>::set_width(width);
	}

private:
	inline size_t size_org_pixel() const { return _org_width*PVBCIBackendImage<Bbits>::height(); }

private:
	pixel_pointer_t _host_img;
	pixel_pointer_t _device_img;
	uint32_t _org_width;
	int _cuda_device;
};

template <size_t Bbits>
struct cuda_kernel;

template <>
struct cuda_kernel<10>
{
	static inline void launch(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream = NULL, bool reverse = false)
	{
		assert(reverse == false);
		(void)reverse; // remove warning "unused parameter"
		show_codes_cuda10(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
	}
};

template <>
struct cuda_kernel<11>
{
	static inline void launch(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream = NULL, bool reverse = false)
	{
		if (reverse) {
			show_codes_cuda11_reverse(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
		}
		else {
			show_codes_cuda11(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
		}
	}
};

class cuda_engine
{
	typedef PVParallelView::PVBCIDrawingBackend<>::render_group_t render_group_t;

	struct device_t
	{
		PVBCICode<>::int_type* device_codes;
		cudaStream_t stream;
	};

public:
	cuda_engine()
	{
		// List all usable cuda engines and create stream and appropriate structure
		PVCuda::visit_usable_cuda_devices([&](int id)
			{
				cudaSetDevice(id);

				device_t dev;
				picviz_verify_cuda(cudaMalloc(&dev.device_codes, NBUCKETS*sizeof(PVBCICode<>::int_type)));
				picviz_verify_cuda(cudaStreamCreate(&dev.stream));
				this->_devices.insert(std::make_pair(id, dev));
			});
		_last_image_dev = _devices.begin();
	}

	~cuda_engine()
	{
		decltype(_devices)::const_iterator it;
		for (it = _devices.begin(); it != _devices.end(); it++) {
			cudaSetDevice(it->first);
			picviz_verify_cuda(cudaStreamDestroy(it->second.stream));
			picviz_verify_cuda(cudaFree(it->second.device_codes));
		}
	}


public:
	void init_engine_thread()
	{
	}

	void free_engine_thread()
	{
	}

	template <size_t Bbits>
	PVBCIBackendImage_p<Bbits> create_image(const size_t img_width) const
	{
		assert(_devices.size() >= 1);
		if (_last_image_dev == _devices.end()) {
			_last_image_dev = _devices.begin();
		}

		int dev = _last_image_dev->first;
		PVBCIBackendImage_p<Bbits> ret(new PVBCIBackendImageCUDA<Bbits>(img_width, dev));
		++_last_image_dev;
		return ret;
	}

	template <size_t Bbits>
	void run(PVBCIBackendImage<Bbits>& dst_img, size_t x_start, size_t width, PVBCICode<Bbits>* codes, size_t n, const float zoom_y, bool reverse, PVBCIDrawingBackend<>::func_cleaning_t cleaning_func, PVBCIDrawingBackend<>::func_drawing_done_t drawing_done, render_group_t const /*g*/)
	{
		typedef PVParallelView::__impl::PVBCIBackendImageCUDA<Bbits> backend_image_t;
#ifdef NBDEUG
		backend_image_t* dst_img_cuda = static_cast<backend_image_t*>(&dst_img);
#else
		backend_image_t* dst_img_cuda = dynamic_cast<backend_image_t*>(&dst_img);
		assert(dst_img_cuda != NULL);
#endif
		assert(x_start + width <= (size_t) dst_img_cuda->width());

		// Set current device according to dst_img
		int dev_id = dst_img_cuda->get_cuda_device();
		dst_img_cuda->set_current_device();
		assert(_devices.find(dev_id) != _devices.end());

		device_t const& dev = _devices[dev_id];

		cudaEvent_t end;
		picviz_verify_cuda(cudaEventCreateWithFlags(&end, cudaEventBlockingSync | cudaEventDisableTiming));

		picviz_verify_cuda(cudaMemcpyAsync(dev.device_codes, codes, n*sizeof(codes), cudaMemcpyHostToDevice, dev.stream));
		cuda_kernel<Bbits>::launch((PVBCICode<Bbits>*)dev.device_codes, n, width, dst_img_cuda->device_img(), dst_img_cuda->org_width(), x_start, zoom_y, dev.stream, reverse);
		dst_img_cuda->copy_device_to_host(dev.stream);
		picviz_verify_cuda(cudaEventRecord(end, dev.stream));
		picviz_verify_cuda(cudaEventSynchronize(end));
		picviz_verify_cuda(cudaEventDestroy(end));

		if (cleaning_func) {
			cleaning_func();
		}
		if (drawing_done) {
			drawing_done();
		}
	}

private:
	std::map<int, device_t> _devices;
	mutable std::map<int, device_t>::const_iterator _last_image_dev;
	cudaStream_t _main_stream;
};

}

class PVBCIDrawingEngineCuda: boost::noncopyable, public PVParallelView::PVBCIDrawingBackendSerial<__impl::cuda_engine>
{
private:
	PVBCIDrawingEngineCuda();

public:
	static PVBCIDrawingEngineCuda* get();
	static void release();

public:
	template <size_t Bbits>
	PVBCIBackendImage_p<Bbits> create_image(const size_t img_width) const
	{
		return __impl::cuda_engine::create_image<Bbits>(img_width);
	}

private:
	static PVBCIDrawingEngineCuda* _s;
	static boost::thread* _thread;
};

template <size_t Bbits = NBITS_INDEX>
class PVBCIDrawingBackendCUDA: public PVBCIDrawingBackend<Bbits>
{
	constexpr static size_t image_height = ((size_t)1)<<Bbits;

	typedef PVParallelView::__impl::PVBCIBackendImageCUDA<Bbits> backend_image_t;
	typedef typename PVBCIDrawingBackend<Bbits>::func_drawing_done_t func_drawing_done_t;
	typedef typename PVBCIDrawingBackend<Bbits>::func_cleaning_t     func_cleaning_t;
	typedef typename PVBCIDrawingBackend<Bbits>::render_group_t render_group_t;

private:
	typedef typename backend_image_t::pixel_t pixel_t;
	typedef typename backend_image_t::pixel_pointer_t pixel_pointer_t;

public:
	PVBCIBackendImage_p<Bbits> create_image(size_t img_width) const
	{
		return PVBCIDrawingEngineCuda::get()->create_image<Bbits>(img_width);
	}

	void operator()(PVBCIBackendImage<Bbits>& dst_img, size_t x_start, size_t width, PVBCICode<Bbits>* codes, size_t n, const float zoom_y = 1.0f, bool reverse = false, func_cleaning_t cleaning_func = func_cleaning_t(), func_drawing_done_t drawing_done = func_drawing_done_t(), render_group_t const rgrp = -1) const override
	{
		PVBCIDrawingEngineCuda::get()->add_job<Bbits>(dst_img, x_start, width, codes, n, zoom_y, reverse, cleaning_func, drawing_done, rgrp);
	}

	virtual render_group_t new_render_group() override { return PVBCIDrawingEngineCuda::get()->new_render_group(); }
	virtual void remove_render_group(render_group_t const g) override { PVBCIDrawingEngineCuda::get()->remove_render_group(g); }
	virtual void cancel_group(render_group_t const g) override { PVBCIDrawingEngineCuda::get()->cancel_group(g); }
};

}

#endif
