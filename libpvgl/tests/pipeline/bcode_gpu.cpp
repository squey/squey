#include <common/common.h>
#include <common/bench.h>

#include <code_bz/types.h>
#include <code_bz/bz_compute.h>
#include <code_bz/serial_bcodecb.h>

#include <picviz/PVPlotted.h>

#include <tbb/pipeline.h>
#include <tbb/task_scheduler_init.h>

#include <cuda/gpu_buf.h>
#include <cuda/gpu_bccb.h>

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*2);
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

typedef std::pair<uint32_t, uint32_t> plotted_range_t;
typedef std::pair<plotted_range_t, GPUPipeBuffer<PVBCode>*> plotted_chunk_t;

class InputPlotted
{
public:
	InputPlotted(Picviz::PVPlotted::plotted_table_t const& table, size_t grain_size, size_t chunk_size, size_t ngpu_buf):
		_grain_size((grain_size*4)/4), _n_buf(ngpu_buf)
	{
		VERIFY(chunk_size % _grain_size == 0);
		VERIFY(chunk_size > _grain_size);
		_gpu_bufs = (GPUPipeBuffer<PVBCode>*) malloc(sizeof(GPUPipeBuffer<PVBCode>)*ngpu_buf);
		for (size_t i = 0; i < ngpu_buf; i++) {
			GPUPipeBuffer<PVBCode>* buf = &_gpu_bufs[i];
			new (buf) GPUPipeBuffer<PVBCode>(grain_size);
			buf->allocate(chunk_size/grain_size, 16);
		}
		_cur_buf = 0;
		_cur_table_row = 0;
		_n_rows = table.size()/2;
	}

	~InputPlotted()
	{
	}

public:
	void free_bufs()
	{
		for (size_t i = 0; i < _n_buf; i++) {
			_gpu_bufs[i].~GPUPipeBuffer<PVBCode>();
		}
		free(_gpu_bufs);
	}

	plotted_chunk_t operator()(tbb::flow_control& fc) const
	{
		if (_cur_table_row >= _n_rows) {
			fc.stop();
			return plotted_chunk_t();
		}

		GPUPipeBuffer<PVBCode>* gpu_buf = &_gpu_bufs[_cur_buf];
		while (!gpu_buf->can_use()) {
			_cur_buf++;
			if (_cur_buf >= _n_buf) {
				_cur_buf = 0;
			}
			gpu_buf = &_gpu_bufs[_cur_buf];
		}
		cudaStream_t stream = gpu_buf->stream();
		if (cudaStreamQuery(stream) == cudaErrorNotReady) {
			fprintf(stderr, "Waiting for GPU to finish with buffer %d...\n", _cur_buf);
			cudaStreamSynchronize(stream);
		}

		fprintf(stderr, "(input) buffer %u is used...\n", _cur_buf);
		size_t end = _cur_table_row + _grain_size;
		if (end > _n_rows) {
			end = _n_rows;
		}
		plotted_chunk_t ret(plotted_range_t(_cur_table_row, end),
		                    gpu_buf);
		_cur_buf++;
		if (_cur_buf >= _n_buf) {
			_cur_buf = 0;
		}
		_cur_table_row = end;

		return ret;
	}

	size_t finish(GPUBccb& gpu_cb)
	{
		size_t ret = 0;
		for (size_t i = 0; i < _n_buf; i++) {
			ret += _gpu_bufs[i].cur();
			gpu_cb.commit_bcode_gpu(_gpu_bufs[i]);
		}
		for (size_t i = 0; i < _n_buf; i++) {
			cudaStreamSynchronize(_gpu_bufs[i].stream());
		}
		return ret;
	}

private:
	GPUPipeBuffer<PVBCode>* _gpu_bufs;
	mutable size_t _cur_table_row;
	size_t _n_rows;
	size_t _grain_size;
	mutable size_t _cur_buf;
	size_t _n_buf;
};

class ProcessPlotted
{
public:
	ProcessPlotted(PVBZCompute& compute):
		_compute(compute)
	{ }
public:
	GPUPipeBuffer<PVBCode>* operator()(plotted_chunk_t chunk) const
	{
		GPUPipeBuffer<PVBCode>* buf = chunk.second;
		printf("Process from %d to %d\n", chunk.first.first, chunk.first.second);
		int ncodes = _compute.compute_b_trans_notable(buf->host()+buf->cur(), chunk.first.first, chunk.first.second, 0, 1);
		if (buf->push_data(ncodes)) {
			printf("Should transfert and maybe launch the kernel.\n");
			return buf; // launch kernel
		}
		buf->free_use();
		return NULL;
	};
private:
	PVBZCompute& _compute;
};

class PushGPU
{
public:
	PushGPU(GPUBccb& bccb, size_t& ncodes, BCodeCB cpu_bccb):
		_bccb(bccb), _ncodes(ncodes), _cpu_bccb(cpu_bccb)
	{
	}
public:
	void operator()(GPUPipeBuffer<PVBCode>* buffer) const
	{
		if (!buffer) {
			return;
		}

		cudaStream_t stream = buffer->stream();
		printf("Transfert buffer...\n");
		buffer->transfert();
		if (buffer->is_full()) {
			printf("Process buffer...\n");
			_ncodes += buffer->cur();
			/*
			PVBCode* tmp = new PVBCode[buffer->cur()];
			verify_cuda(cudaMemcpy(tmp, buffer->dev(), sizeof(PVBCode)*buffer->cur(), cudaMemcpyDeviceToHost));
			serial_bcodecb(tmp, buffer->cur(), _cpu_bccb);
			cudaStreamSynchronize(buffer->stream());
			*/
			cudaStreamSynchronize(buffer->stream());
			// Launch kernel
			_bccb.launch_bcode_gpu(buffer->dev(), buffer->cur(), NULL, stream);
			// And rewind the buffer
			buffer->rewind();
		}
		buffer->free_use();
	}

	GPUBccb& gpu_bccb() { return _bccb; }
	void get_cb(BCodeCB host_cb) const { _bccb.copy_bccb_from_device(host_cb); }
	size_t n_codes() const { return _ncodes; }

private:
	GPUBccb& _bccb;
	BCodeCB _cpu_bccb;
	size_t& _ncodes;
};

#define GRAIN_SIZE (100*1024)
#define CHUNK_SIZE (20*GRAIN_SIZE)
#define NUMBER_CHUNKS 2

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s n\n", argv[0]);
		return 1;
	}

	srand(time(NULL));

	init_cuda();
	std::cout << "Creating random plotted..." << std::endl;
	PVCol ncols = 2;
	Picviz::PVPlotted::plotted_table_t trans_plotted;
	size_t nrows = (atoll(argv[1])/4)*4;
	init_rand_plotted(trans_plotted, nrows);
	std::cout << "Random plotted created." << std::endl;

	PVBZCompute bz;
	bz.set_trans_plotted(trans_plotted, ncols);
	bz.set_zoom(2048, 2048);
	bz.set_box(0, 2048, 0, 2048);

	GPUBccb cuda_bccb;
	BCodeCB cpu_test_bccb = allocate_BCodeCB();
	size_t tmp = 0;
	InputPlotted in_plotted(trans_plotted, GRAIN_SIZE, CHUNK_SIZE, NUMBER_CHUNKS);
	ProcessPlotted process_plotted(bz);
	PushGPU gpu_process(cuda_bccb, tmp, cpu_test_bccb);

	tbb::task_scheduler_init ts_init(1);

	tbb::filter_t<void, plotted_chunk_t> f_in(tbb::filter::serial_in_order, in_plotted);
	tbb::filter_t<plotted_chunk_t, GPUPipeBuffer<PVBCode>*> f_proc(tbb::filter::parallel, process_plotted);
	tbb::filter_t<GPUPipeBuffer<PVBCode>*, void> f_gpu(tbb::filter::serial_in_order, gpu_process);

	tbb::filter_t<void, void> f_pipe = f_in & f_proc & f_gpu;

	std::cout << "Process reduction w/ gpu..." << std::endl;
	tbb::parallel_pipeline(NUMBER_CHUNKS, f_pipe);
	std::cout << "Pipeline finished, synchronized with all streams..." << std::endl;
	size_t n_elts_cuda = in_plotted.finish(gpu_process.gpu_bccb());
	n_elts_cuda += gpu_process.n_codes();
	cudaDeviceSynchronize();
	std::cout << "GPU computation done ! Computing reference..." << std::endl;

	BCodeCB gpu_cb = allocate_BCodeCB();
	gpu_process.get_cb(gpu_cb);

	printf("n_elts cuda: %u\n", n_elts_cuda);

	// Process reference
	BCodeCB cpu_cb = allocate_BCodeCB();
	PVBCode_ap codes;
	posix_memalign((void**) &codes, 16, sizeof(PVBCode)*nrows);
	int ncodes = bz.compute_b_trans_notable(codes, 0, nrows, 0, 1);
	printf("%d\n", ncodes);
	serial_bcodecb(codes, ncodes, cpu_cb);

	CHECK(memcmp(cpu_cb, gpu_cb, SIZE_BCODECB) == 0);

	write(4, cpu_cb, SIZE_BCODECB);
	write(5, gpu_cb, SIZE_BCODECB);
	write(6, cpu_test_bccb, SIZE_BCODECB);

	in_plotted.free_bufs();
	
	return 0;
}
