
// Local copy of PVHitGraphSSEHelpers::buffer_offset_from_y_sse so that each
// translation unit compiles it with its own instruction set.
static inline simde__m128i buffer_offset_local(simde__m128i y_sse,
                                               simde__m128i p_sse,
                                               const simde__m128i y_min_ref_sse,
                                               double alpha,
                                               const simde__m128i zoom_mask_sse,
                                               uint32_t idx_shift,
                                               uint32_t zoom_shift,
                                               size_t nbits)
{
	const simde__m256d alpha_sse = simde_mm256_set1_pd(alpha);

	y_sse = simde_mm_sub_epi32(y_sse, y_min_ref_sse);
	const simde__m256d tmp1_avx = squey_mm256_cvtepu32_pd(y_sse);
	const simde__m256d tmp2_avx = simde_mm256_mul_pd(tmp1_avx, alpha_sse);
	y_sse = squey_mm256_cvttpd_epu32(tmp2_avx);

	p_sse = simde_mm_srli_epi32(y_sse, zoom_shift);
	const simde__m128i off_sse = simde_mm_add_epi32(
		simde_mm_slli_epi32(p_sse, nbits),
		simde_mm_srli_epi32(simde_mm_and_si128(y_sse, zoom_mask_sse), idx_shift)
	);

	return off_sse;
}

static void merge_ctx_buffers(uint32_t* __restrict buffer,
                              PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx,
                              size_t size_int_merge)
{
	size_t packed_size = size_int_merge & ~3;
	size_t j;
	for (j = 0; j < packed_size; j += 4) {
		simde__m128i global_sse = simde_mm_setzero_si128();

		for (int i = 0; i < ctx.get_core_num(); i++) {
			uint32_t* core_buffer = ctx.get_core_buffer(i);
			const simde__m128i local_sse = simde_mm_load_si128((const simde__m128i*)&core_buffer[j]);
			global_sse = simde_mm_add_epi32(global_sse, local_sse);
		}
		simde_mm_storeu_si128((simde__m128i*)&buffer[j], global_sse);
	}
	for (; j < size_int_merge; j++) {
		uint32_t v = 0;
		for (int i = 0; i < ctx.get_core_num(); i++) {
			uint32_t* core_buffer = ctx.get_core_buffer(i);
			v += core_buffer[j];
		}
		buffer[j] = v;
	}
}

//
// OMP algorithms
//

// Optimised version for 1 block, no-selection
void count_y1_omp_sse_v4(const PVRow row_count,
                                const uint32_t* col_y1,
                                const uint64_t y_min,
                                const int zoom,
                                const double alpha,
                                uint32_t* buffer,
                                PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx,
                                size_t nbits,
                                size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const simde__m128i zoom_mask_sse = simde_mm_set1_epi32(zoom_mask);
	const int32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const simde__m128i base_y_sse = simde_mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const simde__m128i y_min_ref_sse = simde_mm_set1_epi32(y_min_ref);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t* my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for (PVRow i = 0; i < packed_row_count; i += 4) {
			const simde__m128i y_sse = simde_mm_load_si128((const simde__m128i*)&col_y1[i]);
			const simde__m128i base_sse = simde_mm_srli_epi32(y_sse, zoom_shift);
			const simde__m128i p_sse = simde_mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if (p < 0)
			 *   continue
			 */
			const simde__m128i res_sse = simde_mm_cmpeq_epi32(p_sse, simde_mm_set1_epi32(0));

			if (simde_mm_test_all_zeros(res_sse, simde_mm_set1_epi32(-1))) {
				continue;
			}

			const simde__m128i off_sse = buffer_offset_local(
			    y_sse, p_sse, y_min_ref_sse, alpha, zoom_mask_sse, idx_shift, zoom_shift,
			    nbits);

			if (simde_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 0)];
			}
			if (simde_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 1)];
			}
			if (simde_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 2)];
			}
			if (simde_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	// last values
	uint32_t* first_buffer = ctx.get_core_buffer(0);
	for (PVRow i = packed_row_count; i < row_count; ++i) {
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y
		// >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int32_t p = base - base_y;
		if (p != 0) {
			continue;
		}
		y = (y - y_min_ref) * alpha;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[idx];
	}

	// final reduction
	size_t merge_size = size_block_int * alpha;
	merge_ctx_buffers(buffer, ctx, merge_size);
}

// Version for N blocks (N >= 2), no-selection
void count_y1_omp_sse_v4(const PVRow row_count,
                                const uint32_t* col_y1,
                                const uint64_t y_min,
                                const int zoom,
                                const double alpha,
                                uint32_t* buffer,
                                int block_count,
                                PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx,
                                size_t nbits,
                                size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const simde__m128i zoom_mask_sse = simde_mm_set1_epi32(zoom_mask);
	const int32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const simde__m128i base_y_sse = simde_mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const simde__m128i y_min_ref_sse = simde_mm_set1_epi32(y_min_ref);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t* my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for (PVRow i = 0; i < packed_row_count; i += 4) {
			const simde__m128i y_sse = simde_mm_load_si128((const simde__m128i*)&col_y1[i]);
			const simde__m128i base_sse = simde_mm_srli_epi32(y_sse, zoom_shift);
			simde__m128i p_sse = simde_mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if ((p < 0) || (p >= block_count))
			 *   continue
			 */
			const simde__m128i res_sse =
			    simde_mm_andnot_si128(simde_mm_cmplt_epi32(p_sse, simde_mm_setzero_si128()),
			                     simde_mm_cmplt_epi32(p_sse, simde_mm_set1_epi32(block_count)));

			if (simde_mm_test_all_zeros(res_sse, simde_mm_set1_epi32(-1))) {
				continue;
			}

			const simde__m128i off_sse = buffer_offset_local(
			    y_sse, p_sse, y_min_ref_sse, alpha, zoom_mask_sse, idx_shift, zoom_shift,
			    nbits);

			if (simde_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 0)];
			}
			if (simde_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 1)];
			}
			if (simde_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 2)];
			}
			if (simde_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	// last values
	uint32_t* first_buffer = ctx.get_core_buffer(0);
	for (PVRow i = packed_row_count; i < row_count; ++i) {
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y
		// >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const int32_t base = (uint64_t)(y) >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		y = (y - y_min_ref) * alpha;
		p = (uint64_t)(y) >> zoom_shift;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[(p << nbits) | idx];
	}

	// final reduction
	size_t merge_size = ((size_t)(size_block_int * alpha)) * block_count;
	merge_ctx_buffers(buffer, ctx, merge_size);
}

// Version for N blocks (N>=1), with selection
void count_y1_sel_omp_sse_v4(const PVRow row_count,
                             const uint32_t* col_y1,
                             const Squey::PVSelection& selection,
                             const uint64_t y_min,
                             const int zoom,
                             const double& alpha,
                             uint32_t* buffer,
                             int block_count,
                             PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx,
                             size_t nbits,
                             size_t size_block_int)
{
	static DECLARE_ALIGN(16) simde__m128i mask[16] = {
	    simde_mm_set_epi32(0, 0, 0, 0),    simde_mm_set_epi32(0, 0, 0, -1),   simde_mm_set_epi32(0, 0, -1, 0),
	    simde_mm_set_epi32(0, 0, -1, -1),  simde_mm_set_epi32(0, -1, 0, 0),   simde_mm_set_epi32(0, -1, 0, -1),
	    simde_mm_set_epi32(0, -1, -1, 0),  simde_mm_set_epi32(0, -1, -1, -1), simde_mm_set_epi32(-1, 0, 0, 0),
	    simde_mm_set_epi32(-1, 0, 0, -1),  simde_mm_set_epi32(-1, 0, -1, 0),  simde_mm_set_epi32(-1, 0, -1, -1),
	    simde_mm_set_epi32(-1, -1, 0, 0),  simde_mm_set_epi32(-1, -1, 0, -1), simde_mm_set_epi32(-1, -1, -1, 0),
	    simde_mm_set_epi32(-1, -1, -1, -1)};

	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const simde__m128i zoom_mask_sse = simde_mm_set1_epi32(zoom_mask);
	const uint32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const simde__m128i base_y_sse = simde_mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const simde__m128i y_min_ref_sse = simde_mm_set1_epi32(y_min_ref);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t* my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for (PVRow i = 0; i < packed_row_count; i += 4) {
			uint32_t f = selection.get_lines_fast(i, 4);
			if (f == 0) {
				continue;
			}

			simde__m128i y_sse = simde_mm_load_si128((const simde__m128i*)&col_y1[i]);
			const simde__m128i base_sse = simde_mm_srli_epi32(y_sse, zoom_shift);
			simde__m128i p_sse = simde_mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if (!sel.is_set(y) && (p < 0) || (p >= block_count))
			 *   continue
			 */
			const simde__m128i res_sse =
			    simde_mm_and_si128(simde_mm_andnot_si128(simde_mm_cmplt_epi32(p_sse, simde_mm_setzero_si128()),
			                                   simde_mm_cmplt_epi32(p_sse, simde_mm_set1_epi32(block_count))),
			                  mask[f]);
			if (simde_mm_test_all_zeros(res_sse, simde_mm_set1_epi32(-1))) {
				continue;
			}

			const simde__m128i off_sse = buffer_offset_local(
			    y_sse, p_sse, y_min_ref_sse, alpha, zoom_mask_sse, idx_shift, zoom_shift,
			    nbits);

			if (simde_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 0)];
			}
			if (simde_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 1)];
			}
			if (simde_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 2)];
			}
			if (simde_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[simde_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	uint32_t* first_buffer = ctx.get_core_buffer(0);
	for (PVRow i = packed_row_count; i < row_count; ++i) {
		if (!selection.get_line_fast(i)) {
			continue;
		}
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y
		// >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const uint32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		y = (y - y_min_ref) * alpha;
		p = y >> zoom_shift;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[(p << nbits) + idx];
	}

	// final reduction
	size_t merge_size = size_block_int * block_count * alpha;
	merge_ctx_buffers(buffer, ctx, merge_size);
}