/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H

#include <pvkernel/cuda/common.h>

#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/concurrent_queue.h>

#include <QImage>

#include <map>

#include <cassert>

namespace PVParallelView
{

class PVBCIDrawingBackendCUDA;

/**
 * It is an image convertible to QImage with host/device management.
 *
 * Calls for data transfert have to be done manualy to make sure they are done
 * the right time
 * FIXME : Check this assert.
 */
class PVBCIBackendImageCUDA : public PVParallelView::PVBCIBackendImage
{
	friend class PVParallelView::PVBCIDrawingBackendCUDA;
	friend class cuda_engine;

	using pixel_t = uint32_t;

  public:
	PVBCIBackendImageCUDA(const uint32_t width,
	                      uint8_t height_bits,
	                      const int cuda_device,
	                      cudaStream_t stream);
	~PVBCIBackendImageCUDA() override;

	// AG: should be protected, but used directly in some benchmarking tests
  public:
	pixel_t* device_img() { return _device_img; }
	pixel_t* host_img() { return _host_img; }

	pixel_t const* device_img() const { return _device_img; }
	pixel_t const* host_img() const { return _host_img; }

	inline void copy_device_to_host(cudaStream_t const& stream) const
	{
		inendi_verify_cuda(cudaMemcpyAsync(_host_img, _device_img,
		                                   size_org_pixel() * sizeof(pixel_t),
		                                   cudaMemcpyDeviceToHost, stream));
	}

	inline void copy_device_to_host() const
	{
		inendi_verify_cuda(cudaMemcpy(_host_img, _device_img, size_org_pixel() * sizeof(pixel_t),
		                              cudaMemcpyDeviceToHost));
	}

	inline size_t org_width() const { return _org_width; }

	void set_current_device() const { inendi_verify_cuda(cudaSetDevice(_cuda_device)); }

	int get_cuda_device() const { return _cuda_device; }

  public:
	virtual QImage qimage(size_t crop_height) const
	{
		assert(crop_height <= PVBCIBackendImage::height());
		// This is really important to use the const constructor version of QImage, so that no data
		// copy is made !
		return QImage((const uchar*)_host_img, PVBCIBackendImage::width(), crop_height,
		              _org_width * sizeof(uint32_t), QImage::Format_ARGB32_Premultiplied);
	}

	virtual bool set_width(uint32_t width)
	{
		if (width > _org_width) {
			return false;
		}

		return PVBCIBackendImage::set_width(width);
	}

  private:
	inline size_t size_org_pixel() const { return _org_width * PVBCIBackendImage::height(); }

  private:
	pixel_t* _host_img;   //!< Pointer to image representation on the host
	pixel_t* _device_img; //!< Pointer to image representation on the device.
	uint32_t _org_width;  //!< Allocatad with size on the host.
	int _cuda_device;     //!< Id of the cuda device use for computation.
};

class PVBCIDrawingBackendCUDA : public PVBCIDrawingBackendAsync
{
	typedef PVBCIBackendImageCUDA backend_image_t;

	struct cuda_job_data {
		std::function<void()> done_function;
	};

	struct device_t {
		PVBCICodeBase* device_codes;
		cudaStream_t stream;
	};

  private:
	PVBCIDrawingBackendCUDA();

  public:
	virtual ~PVBCIDrawingBackendCUDA();
	static PVBCIDrawingBackendCUDA& get();

  public:
	Flags flags() const { return Serial; }
	PVBCIBackendImage_p create_image(size_t img_width, uint8_t height_bits) override;

  public:
	PVBCICodeBase* allocate_bci(size_t n) override;
	void free_bci(PVBCICodeBase* buf) override;

  public:
	void operator()(PVBCIBackendImage_p& dst_img,
	                size_t x_start,
	                size_t width,
	                PVBCICodeBase* codes,
	                size_t n,
	                const float zoom_y = 1.0f,
	                bool reverse = false,
	                std::function<void()> const& render_done = std::function<void()>()) override;
	void wait_all() const;

  private:
	/**
	 * Callback function called once image creation is done and back on computer.
	 */
	static void
	image_rendered_and_copied_callback(cudaStream_t stream, cudaError_t status, void* data);

  private:
	std::map<int, device_t> _devices; //!< List of available device from their ids
	std::map<int, device_t>::const_iterator
	    _last_image_dev; //!< Next device to use for computation.
};
}

#endif
