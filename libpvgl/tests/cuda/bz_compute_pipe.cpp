#include <code_bz/bz_compute.h>
#include <cassert>

#include <cuda/gpu_bccb.h>
#include <cuda/gpu_buf.h>

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/picviz_libdivide.h>

#include <omp.h>
#include <cuda/common.h>

#define NTMP_BUF 12

int PVBZCompute::compute_b_trans_sse4_notable_omp_pipe(PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1, size_t size_chunk, GPUBccb& gpu_bccb, int nthreads)
{
	// It is assumed that codes is a pointer to a list of buffers of size
	// nrows/nthreads.
	
	if (size_chunk % 4 != 0) {
		return -1;
	}
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X1, Y1));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	x1 -= (int)x0;
	float Xnorm = x0*_zoom_x;

	__m128 sse_x0 = _mm_set1_ps(x0);
	__m128 sse_x0comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x0));
	__m128 sse_x1 = _mm_set1_ps(x1);
	__m128 sse_x1comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x1));
	__m128 sse_y0 = _mm_set1_ps(y0);
	__m128 sse_y1 = _mm_set1_ps(y1);

	__m128i sse_Y0 = _mm_set1_epi32(Y0);
	__m128i sse_Xnorm = _mm_set1_epi32((int) Xnorm);

	__m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	__m128 sse_zoomy = _mm_set1_ps(_zoom_y);

	int n_codes = 0;
	double time_taken = 0;
#pragma omp parallel reduction(+:n_codes) firstprivate(sse_x0, sse_x0comp, sse_x1, sse_x1comp, sse_y0, sse_y1, sse_Y0, sse_Xnorm, sse_zoomx, sse_zoomy) num_threads(nthreads)
	{
		// Init our double-buffered pipe
		PVBCode* buf_codes[NTMP_BUF];
		PVBCode* buf_dev_codes[NTMP_BUF];
		PVBCode* buf_free[NTMP_BUF];
		cudaEvent_t events[NTMP_BUF];
		for (int i = 0; i < NTMP_BUF; i++) {
			buf_codes[i] = gpu_bccb.allocate_host_bcode_buffer(size_chunk, &buf_dev_codes[i], &buf_free[i]);
			events[i] = NULL;
		}
		PVBCode_ap codes = (PVBCode_ap) buf_codes[0];

		int cur_buf_codes = 0;
		__m128i sse_1i = _mm_set1_epi32(1);

		const PVRow offa = axis_a*_nb_rows;
		const PVRow offb = axis_b*_nb_rows;
		const PVRow end = (_nb_rows*size_chunk)/size_chunk;

		tbb::tick_count start = tbb::tick_count::now();

#pragma omp barrier

#pragma omp for schedule(static)
		for (PVRow c = 0; c < end; c += size_chunk) {
			const PVRow end_c = c+size_chunk;
			int idx_code = 0;

			// Wait for the current event if necessary before going on
			if (events[cur_buf_codes]) {
				//if (cudaEventQuery(events[cur_buf_codes]) == cudaErrorNotReady) {
				//	printf("waiting for gpu !!\n");
					verify_cuda(cudaEventSynchronize(events[cur_buf_codes]));
				//}
				verify_cuda(cudaEventDestroy(events[cur_buf_codes]));
				events[cur_buf_codes] = 0;
			}

			for (PVRow i = c; i < end_c; i += 4) {
				__m128 sse_ypl,sse_ypr;
				sse_ypl = _mm_load_ps(&_trans_plotted[offa+i]);
				sse_ypr = _mm_load_ps(&_trans_plotted[offb+i]);

				// Line equation
				// a*X + b*Y + c = 0
				//
				// (ypl-ypr)*x + y - ypl = 0
				// or...
				// (ypl-ypr)*x + y >= ypl

				const __m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
				const __m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), sse_1i);
				const __m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), sse_1i);
				const __m128i c = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl));
				const __m128i d = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl));
				// r0 = a & !d
				// r1 = b & !c
				// r2 = c | d
				const __m128i r0 = _mm_and_si128(a, _mm_andnot_si128(d, sse_1i));
				const __m128i r1 = _mm_and_si128(b, _mm_andnot_si128(c, sse_1i));
				const __m128i r2 = _mm_and_si128(_mm_or_si128(c, d), sse_1i);
				// types = r0 | r1 << 1 | r2 << 2
				const __m128i sse_types = _mm_or_si128(r0, _mm_or_si128(_mm_slli_epi32(r1, 1),
							_mm_slli_epi32(r2, 2)));

				// Special cases when all the types are the same. Let's do this in SSE !!
				int64_t dtypes_0 = _mm_extract_epi64(sse_types, 0);
				int64_t dtypes_1 = _mm_extract_epi64(sse_types, 1);
				if (dtypes_0 == dtypes_1) {
					// Take the first type.
					const uint32_t type = _mm_extract_epi32(sse_types, 0);
					__m128i sse_bcodes_li, sse_bcodes_ri;
					__m128 tmpl,tmpr;
					switch (type) {
						case 0:
							continue;
						case 5:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 3:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
						case 1:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 2:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
						case 4:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 6:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
					}
					__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
							_mm_slli_epi32(sse_bcodes_ri, 14));
					__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

					if ((idx_code & 3) == 0) {
						// We are still 16-byte aligned
						_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
					else {
						// Un-aligned store, performance loss... :s
						// Check the difference between this and 4 _mm_stream_si32 !
						_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}

					continue;
				}

				// Check if one of the type is 0.
				// If not, using some SSE optimisations !
				if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFF))) {
					__m128 sse_bcodes_l;
					__m128 sse_bcodes_r;
					for (int j = 0; j < 4; j++) {
						float ypl,ypr,fydiff;
						_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
						_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
						_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
						union { int i; float f; } bcode_l, bcode_r;
						int type = _mm_extract_epi32(sse_types, j);
						switch (type) {
							case 5:
								bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 3:
								bcode_l.f = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
								bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 1:
								bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r.f = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 2:
								bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 4:
								bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 6:
								bcode_l.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
								break;
#ifndef NDEBUG
							default:
								assert(false);
								break;
#endif
						}
						// _mm_insert_ps does not suit our purpose.
						// This will not work without optimisations, because this loop won't be unrolled,
						// and _mm_insert_epi32 waits for a constant !
						sse_bcodes_l = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_l), bcode_l.i, j));
						sse_bcodes_r = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_r), bcode_r.i, j));
					}
					// Use sse to set the codes. Casting is also done in SSE, which gives different
					// results that the serial one (+/- 1 !).
					// Format:
					//   * 3 bits: types
					//   * 11 bits: l
					//   * 11 bits: r
					//   * free = 0
					__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_l), 3),
							_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_r), 14));
					__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

					if ((idx_code & 3) == 0) {
						// We are style 16-byte aligned
						_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
					else {
						// Un-aligned store, performance loss... :s
						// Check the difference between this and 4 _mm_stream_si32 !
						_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
				}
				else {
					// No SSE possible here, do it by hand.
					PVBCode bcode;
					bcode.int_v = 0;
					for (int j = 0; j < 4; j++) {
						float ypl,ypr,fydiff,bcode_l,bcode_r;
						_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
						_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
						_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
						int type = _mm_extract_epi32(sse_types, j);
						switch (type) {
							case 0:
								continue;
							case 5:
								bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 3:
								bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
								bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 1:
								bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 2:
								bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 4:
								bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 6:
								bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
								break;
							default:
								assert(false);
								break;
						}
						bcode.s.type = type;
						bcode.s.l = (uint16_t) bcode_l;
						bcode.s.r = (uint16_t) bcode_r;

						codes[idx_code] = bcode;
						idx_code++;
					}
				}
			}
			// Transfert
			gpu_bccb.push_bcode_gpu(codes, buf_dev_codes[cur_buf_codes], idx_code, &events[cur_buf_codes]);

			// Double buffering, use the other buffer
			cur_buf_codes = (cur_buf_codes+1)%NTMP_BUF;
			codes = buf_codes[cur_buf_codes];

			n_codes += idx_code;
		}
		tbb::tick_count time_end = tbb::tick_count::now();
#pragma omp critical
		{
			double time = (time_end-start).seconds();
			if (time > time_taken) {
				time_taken = time;
			}
		}

		// Free CUDA ressources
		for (int i = 0; i < NTMP_BUF; i++) {
			gpu_bccb.free_host_bcode_buffer(buf_free[i], buf_dev_codes[i]);
		}
	}
	printf("Time taken: %0.4f ms\n", time_taken*1000.0);

	return n_codes;
}

#undef NTMP_BUF
#define NTMP_BUF (6)
#define NCHUNKS_TMP_BUF (1000)

int PVBZCompute::compute_b_trans_sse4_notable_omp_pipe2(PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1, size_t size_chunk, GPUBccb& gpu_bccb, int nthreads)
{
	// It is assumed that codes is a pointer to a list of buffers of size
	// nrows/nthreads.
	
	if (size_chunk % 4 != 0) {
		return -1;
	}
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X1, Y1));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	x1 -= (int)x0;
	float Xnorm = x0*_zoom_x;

	__m128 sse_x0 = _mm_set1_ps(x0);
	__m128 sse_x0comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x0));
	__m128 sse_x1 = _mm_set1_ps(x1);
	__m128 sse_x1comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x1));
	__m128 sse_y0 = _mm_set1_ps(y0);
	__m128 sse_y1 = _mm_set1_ps(y1);

	__m128i sse_Y0 = _mm_set1_epi32(Y0);
	__m128i sse_Xnorm = _mm_set1_epi32((int) Xnorm);

	__m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	__m128 sse_zoomy = _mm_set1_ps(_zoom_y);

	int n_codes = 0;
	double time_taken = 0;
#pragma omp parallel reduction(+:n_codes) firstprivate(sse_x0, sse_x0comp, sse_x1, sse_x1comp, sse_y0, sse_y1, sse_Y0, sse_Xnorm, sse_zoomx, sse_zoomy) num_threads(nthreads)
	{
		// Init our double-buffered pipe
		GPUPipeBuffer<PVBCode>* buf_codes[NTMP_BUF];
		//cudaEvent_t events[NTMP_BUF];
		for (int i = 0; i < NTMP_BUF; i++) {
			buf_codes[i] = new GPUPipeBuffer<PVBCode>(size_chunk);
			buf_codes[i]->allocate(NCHUNKS_TMP_BUF, 16);
			//events[i] = NULL;
		}
		PVBCode_ap codes = (PVBCode_ap) buf_codes[0]->host();

		int cur_buf_codes = 0;
		__m128i sse_1i = _mm_set1_epi32(1);

		const PVRow offa = axis_a*_nb_rows;
		const PVRow offb = axis_b*_nb_rows;
		const PVRow end = (_nb_rows*size_chunk)/size_chunk;

		//cudaStream_t stream;
		//verify_cuda(cudaStreamCreate(&stream));
		tbb::tick_count start = tbb::tick_count::now();

#pragma omp barrier

		int idx_code = 0;
		int nb_push = 0;
		int nb_commit = 0;
#pragma omp for schedule(guided)
		for (PVRow c = 0; c < end; c += size_chunk) {
			const PVRow end_c = c+size_chunk;

			for (PVRow i = c; i < end_c; i += 4) {
				__m128 sse_ypl,sse_ypr;
				sse_ypl = _mm_load_ps(&_trans_plotted[offa+i]);
				sse_ypr = _mm_load_ps(&_trans_plotted[offb+i]);

				// Line equation
				// a*X + b*Y + c = 0
				//
				// (ypl-ypr)*x + y - ypl = 0
				// or...
				// (ypl-ypr)*x + y >= ypl

				const __m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
				const __m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), sse_1i);
				const __m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), sse_1i);
				const __m128i c = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl));
				const __m128i d = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl));
				// r0 = a & !d
				// r1 = b & !c
				// r2 = c | d
				const __m128i r0 = _mm_and_si128(a, _mm_andnot_si128(d, sse_1i));
				const __m128i r1 = _mm_and_si128(b, _mm_andnot_si128(c, sse_1i));
				const __m128i r2 = _mm_and_si128(_mm_or_si128(c, d), sse_1i);
				// types = r0 | r1 << 1 | r2 << 2
				const __m128i sse_types = _mm_or_si128(r0, _mm_or_si128(_mm_slli_epi32(r1, 1),
							_mm_slli_epi32(r2, 2)));

				// Special cases when all the types are the same. Let's do this in SSE !!
				int64_t dtypes_0 = _mm_extract_epi64(sse_types, 0);
				int64_t dtypes_1 = _mm_extract_epi64(sse_types, 1);
				if (dtypes_0 == dtypes_1) {
					// Take the first type.
					const uint32_t type = _mm_extract_epi32(sse_types, 0);
					__m128i sse_bcodes_li, sse_bcodes_ri;
					__m128 tmpl,tmpr;
					switch (type) {
						case 0:
							continue;
						case 5:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 3:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
						case 1:
							tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 2:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
						case 4:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							break;
						case 6:
							tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);
							tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

							sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
							sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
							break;
					}
					__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
							_mm_slli_epi32(sse_bcodes_ri, 14));
					__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

					if ((idx_code & 3) == 0) {
						// We are still 16-byte aligned
						_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
					else {
						// Un-aligned store, performance loss... :s
						// Check the difference between this and 4 _mm_stream_si32 !
						_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}

					continue;
				}

				// Check if one of the type is 0.
				// If not, using some SSE optimisations !
				if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFF))) {
					__m128 sse_bcodes_l;
					__m128 sse_bcodes_r;
					for (int j = 0; j < 4; j++) {
						float ypl,ypr,fydiff;
						_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
						_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
						_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
						union { int i; float f; } bcode_l, bcode_r;
						int type = _mm_extract_epi32(sse_types, j);
						switch (type) {
							case 5:
								bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 3:
								bcode_l.f = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
								bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 1:
								bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r.f = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 2:
								bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 4:
								bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 6:
								bcode_l.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								bcode_r.f = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
								break;
#ifndef NDEBUG
							default:
								assert(false);
								break;
#endif
						}
						// _mm_insert_ps does not suit our purpose.
						// This will not work without optimisations, because this loop won't be unrolled,
						// and _mm_insert_epi32 waits for a constant !
						sse_bcodes_l = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_l), bcode_l.i, j));
						sse_bcodes_r = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_r), bcode_r.i, j));
					}
					// Use sse to set the codes. Casting is also done in SSE, which gives different
					// results that the serial one (+/- 1 !).
					// Format:
					//   * 3 bits: types
					//   * 11 bits: l
					//   * 11 bits: r
					//   * free = 0
					__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_l), 3),
							_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_r), 14));
					__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

					if ((idx_code & 3) == 0) {
						// We are style 16-byte aligned
						_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
					else {
						// Un-aligned store, performance loss... :s
						// Check the difference between this and 4 _mm_stream_si32 !
						_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
						idx_code += 4;
					}
				}
				else {
					// No SSE possible here, do it by hand.
					PVBCode bcode;
					bcode.int_v = 0;
					for (int j = 0; j < 4; j++) {
						float ypl,ypr,fydiff,bcode_l,bcode_r;
						_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
						_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
						_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
						int type = _mm_extract_epi32(sse_types, j);
						switch (type) {
							case 0:
								continue;
							case 5:
								bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 3:
								bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
								bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 1:
								bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
								bcode_r = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 2:
								bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
								break;
							case 4:
								bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								break;
							case 6:
								bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
								bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
								break;
							default:
								assert(false);
								break;
						}
						bcode.s.type = type;
						bcode.s.l = (uint16_t) bcode_l;
						bcode.s.r = (uint16_t) bcode_r;

						codes[idx_code] = bcode;
						idx_code++;
					}
				}
			}
			// Transfert
			//if (gpu_bccb.push_bcode_gpu(*buf_codes[cur_buf_codes], idx_code, &events[cur_buf_codes])) {
			nb_push++;
			if (gpu_bccb.push_bcode_gpu(*buf_codes[cur_buf_codes], idx_code-buf_codes[cur_buf_codes]->cur(), NULL)) {
				// Double buffering, use the other buffer
				nb_commit++;
				cur_buf_codes = (cur_buf_codes+1)%NTMP_BUF;
				codes = buf_codes[cur_buf_codes]->host();

				// Wait for the current event if necessary before going on
				/*
				if (events[cur_buf_codes]) {
					if (cudaEventQuery(events[cur_buf_codes]) == cudaErrorNotReady) {
						printf("waiting for gpu !!\n");
						verify_cuda(cudaEventSynchronize(events[cur_buf_codes]));
					}
					verify_cuda(cudaEventDestroy(events[cur_buf_codes]));
					events[cur_buf_codes] = 0;
				}*/
				cudaStream_t s = buf_codes[cur_buf_codes]->stream();
				if (cudaStreamQuery(s) == cudaErrorNotReady) {
					printf("waiting for gpu !!\n");
					verify_cuda(cudaStreamSynchronize(s));
				}
				n_codes += idx_code;
				idx_code = 0;
			}
		}
		tbb::tick_count time_end = tbb::tick_count::now();
		for (int i = 0; i < NTMP_BUF; i++) {
			/*if (events[i]) {
				verify_cuda(cudaEventSynchronize(events[i]));
				verify_cuda(cudaEventDestroy(events[i]));
			}*/
			verify_cuda(cudaStreamSynchronize(buf_codes[i]->stream()));
			gpu_bccb.commit_bcode_gpu_and_wait(*buf_codes[i]);
		}
		printf("Nb push: %d | Nb commits: %d\n", nb_push, nb_commit);
		//cudaStreamDestroy(stream);
#pragma omp critical
		{
			double time = (time_end-start).seconds();
			if (time > time_taken) {
				time_taken = time;
			}
		}
	}
	printf("Time taken: %0.4f ms\n", time_taken*1000.0);

	return n_codes;
}

int PVBZCompute::compute_b_trans_sse4_notable_omp_pipe4(PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1, size_t size_chunk, GPUBccb& gpu_bccb, int nthreads)
{
	// It is assumed that codes is a pointer to a list of buffers of size
	// nrows/nthreads.
	
	if (size_chunk % 4 != 0) {
		return -1;
	}
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X1, Y1));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	x1 -= (int)x0;
	float Xnorm = x0*_zoom_x;

	__m128 sse_x0 = _mm_set1_ps(x0);
	__m128 sse_x0comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x0));
	__m128 sse_x1 = _mm_set1_ps(x1);
	__m128 sse_x1comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x1));
	__m128 sse_y0 = _mm_set1_ps(y0);
	__m128 sse_y1 = _mm_set1_ps(y1);

	__m128i sse_Y0 = _mm_set1_epi32(Y0);
	__m128i sse_Xnorm = _mm_set1_epi32((int) Xnorm);

	__m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	__m128 sse_zoomy = _mm_set1_ps(_zoom_y);

	int n_codes = 0;
	double time_taken = 0;
#pragma omp parallel reduction(+:n_codes) firstprivate(sse_x0, sse_x0comp, sse_x1, sse_x1comp, sse_y0, sse_y1, sse_Y0, sse_Xnorm, sse_zoomx, sse_zoomy) num_threads(nthreads)
	{
		// Init our double-buffered pipe
		GPUPipeBuffer<PVBCode>* buf_codes[NTMP_BUF];
		//cudaEvent_t events[NTMP_BUF];
		for (int i = 0; i < NTMP_BUF; i++) {
			buf_codes[i] = new GPUPipeBuffer<PVBCode>(4);
			buf_codes[i]->allocate(_nb_rows/4, 16);
			//events[i] = NULL;
		}
		PVBCode_ap codes = (PVBCode_ap) buf_codes[0]->host();

		int cur_buf_codes = 0;
		__m128i sse_1i = _mm_set1_epi32(1);

		const PVRow offa = axis_a*_nb_rows;
		const PVRow offb = axis_b*_nb_rows;
		const PVRow end = (_nb_rows*4)/4;

		//cudaStream_t stream;
		//verify_cuda(cudaStreamCreate(&stream));
		tbb::tick_count start = tbb::tick_count::now();

#pragma omp barrier

		int idx_code = 0;
		int nb_push = 0;
		int nb_commit = 0;
#pragma omp for schedule(guided)
		for (PVRow i = 0; i < end; i += 4) {
			__m128 sse_ypl,sse_ypr;
			sse_ypl = _mm_load_ps(&_trans_plotted[offa+i]);
			sse_ypr = _mm_load_ps(&_trans_plotted[offb+i]);

			// Line equation
			// a*X + b*Y + c = 0
			//
			// (ypl-ypr)*x + y - ypl = 0
			// or...
			// (ypl-ypr)*x + y >= ypl

			const __m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
			const __m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), sse_1i);
			const __m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), sse_1i);
			const __m128i c = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl));
			const __m128i d = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl));
			// r0 = a & !d
			// r1 = b & !c
			// r2 = c | d
			const __m128i r0 = _mm_and_si128(a, _mm_andnot_si128(d, sse_1i));
			const __m128i r1 = _mm_and_si128(b, _mm_andnot_si128(c, sse_1i));
			const __m128i r2 = _mm_and_si128(_mm_or_si128(c, d), sse_1i);
			// types = r0 | r1 << 1 | r2 << 2
			const __m128i sse_types = _mm_or_si128(r0, _mm_or_si128(_mm_slli_epi32(r1, 1),
						_mm_slli_epi32(r2, 2)));

			// Special cases when all the types are the same. Let's do this in SSE !!
			int64_t dtypes_0 = _mm_extract_epi64(sse_types, 0);
			int64_t dtypes_1 = _mm_extract_epi64(sse_types, 1);
			if (dtypes_0 == dtypes_1) {
				// Take the first type.
				const uint32_t type = _mm_extract_epi32(sse_types, 0);
				__m128i sse_bcodes_li, sse_bcodes_ri;
				__m128 tmpl,tmpr;
				switch (type) {
					case 0:
						continue;
					case 5:
						tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
						tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						break;
					case 3:
						tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
						tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
						break;
					case 1:
						tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
						tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						break;
					case 2:
						tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
						tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
						break;
					case 4:
						tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
						tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						break;
					case 6:
						tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);
						tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

						sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
						sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
						break;
				}
				__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
						_mm_slli_epi32(sse_bcodes_ri, 14));
				__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

				if ((idx_code & 3) == 0) {
					// We are still 16-byte aligned
					_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
					idx_code += 4;
				}
				else {
					// Un-aligned store, performance loss... :s
					// Check the difference between this and 4 _mm_stream_si32 !
					_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
					idx_code += 4;
				}

				continue;
			}

			// Check if one of the type is 0.
			// If not, using some SSE optimisations !
			if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFF))) {
				__m128 sse_bcodes_l;
				__m128 sse_bcodes_r;
				for (int j = 0; j < 4; j++) {
					float ypl,ypr,fydiff;
					_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
					_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
					_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
					union { int i; float f; } bcode_l, bcode_r;
					int type = _mm_extract_epi32(sse_types, j);
					switch (type) {
						case 5:
							bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
							bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 3:
							bcode_l.f = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
							bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
							break;
						case 1:
							bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
							bcode_r.f = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 2:
							bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
							bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
							break;
						case 4:
							bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
							bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 6:
							bcode_l.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							bcode_r.f = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
							break;
#ifndef NDEBUG
						default:
							assert(false);
							break;
#endif
					}
					// _mm_insert_ps does not suit our purpose.
					// This will not work without optimisations, because this loop won't be unrolled,
					// and _mm_insert_epi32 waits for a constant !
					sse_bcodes_l = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_l), bcode_l.i, j));
					sse_bcodes_r = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_r), bcode_r.i, j));
				}
				// Use sse to set the codes. Casting is also done in SSE, which gives different
				// results that the serial one (+/- 1 !).
				// Format:
				//   * 3 bits: types
				//   * 11 bits: l
				//   * 11 bits: r
				//   * free = 0
				__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_l), 3),
						_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_r), 14));
				__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);

				if ((idx_code & 3) == 0) {
					// We are style 16-byte aligned
					_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
					idx_code += 4;
				}
				else {
					// Un-aligned store, performance loss... :s
					// Check the difference between this and 4 _mm_stream_si32 !
					_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
					idx_code += 4;
				}
			}
			else {
				// No SSE possible here, do it by hand.
				PVBCode bcode;
				bcode.int_v = 0;
				for (int j = 0; j < 4; j++) {
					float ypl,ypr,fydiff,bcode_l,bcode_r;
					_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
					_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
					_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
					int type = _mm_extract_epi32(sse_types, j);
					switch (type) {
						case 0:
							continue;
						case 5:
							bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
							bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 3:
							bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
							bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
							break;
						case 1:
							bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
							bcode_r = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 2:
							bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
							bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
							break;
						case 4:
							bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
							bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							break;
						case 6:
							bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
							bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
							break;
						default:
							assert(false);
							break;
					}
					bcode.s.type = type;
					bcode.s.l = (uint16_t) bcode_l;
					bcode.s.r = (uint16_t) bcode_r;

					codes[idx_code] = bcode;
					idx_code++;
				}
			}
#if 0
			size_t new_codes = idx_code-buf_codes[cur_buf_codes]->cur();
			if (new_codes > 0 && gpu_bccb.push_bcode_gpu(*buf_codes[cur_buf_codes], new_codes, NULL)) {
				// Double buffering, use the other buffer
				cur_buf_codes = (cur_buf_codes+1)%NTMP_BUF;
				codes = buf_codes[cur_buf_codes]->host();

				// Wait for the current event if necessary before going on
				/*
				   if (events[cur_buf_codes]) {
				   if (cudaEventQuery(events[cur_buf_codes]) == cudaErrorNotReady) {
				   printf("waiting for gpu !!\n");
				   verify_cuda(cudaEventSynchronize(events[cur_buf_codes]));
				   }
				   verify_cuda(cudaEventDestroy(events[cur_buf_codes]));
				   events[cur_buf_codes] = 0;
				   }*/
				cudaStream_t s = buf_codes[cur_buf_codes]->stream();
				if (cudaStreamQuery(s) == cudaErrorNotReady) {
					printf("waiting for gpu !!\n");
					verify_cuda(cudaStreamSynchronize(s));
				}
				n_codes += idx_code;
				idx_code = 0;
			}
#endif
		}
		tbb::tick_count time_end = tbb::tick_count::now();
		for (int i = 0; i < NTMP_BUF; i++) {
			/*if (events[i]) {
			  verify_cuda(cudaEventSynchronize(events[i]));
			  verify_cuda(cudaEventDestroy(events[i]));
			  }*/
			//verify_cuda(cudaStreamSynchronize(buf_codes[i]->stream()));
			//gpu_bccb.commit_bcode_gpu_and_wait(*buf_codes[i]);
		}
		//cudaStreamDestroy(stream);
#pragma omp critical
		{
			double time = (time_end-start).seconds();
			if (time > time_taken) {
				time_taken = time;
			}
		}
	}
	printf("Time taken: %0.4f ms\n", time_taken*1000.0);

	return n_codes;
}
