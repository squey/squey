__kernel void bccb2(__global const unsigned int* bcodes, const unsigned int n, unsigned int offset, __global unsigned int* bccb)
{
	const unsigned int lidx = get_local_id(0);
	const unsigned int bccb_idx_x = get_group_id(0)*get_local_size(0)+lidx;

	__global unsigned int* bccb_item = &bccb[bccb_idx_x];
	__local unsigned int bcodes_local[256*12];
	offset = 256*12;

	const unsigned int nend = (n/offset)*offset;
	unsigned int code, code_idx;
	unsigned int bccb_v = 0;
	for (unsigned int off = 0; off < nend; off += offset) {
		for (unsigned int i = 0; i < offset; i++) {
			bcodes_local[i] = bcodes[off+i];
			barrier(CLK_LOCAL_MEM_FENCE);
			code = bcodes_local[i];
			code_idx = code>>5;
			if (code_idx == bccb_idx_x) {
				bccb_v |= (1<<(code&31));
				if (bccb_v == 0xFFFFFFFF) {
					// Commit to global mem
					*bccb_item = 0xFFFFFFFF;
					return;
				}
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	for (unsigned int i = nend; i < n; i++) {
		code = bcodes[i];
		code_idx = code>>5;
		if (code_idx == bccb_idx_x) {
			bccb_v |= (1<<(code&31));
			if (bccb_v == 0xFFFFFFFF) {
				// Commit to global mem
				*bccb_item = 0xFFFFFFFF;
				return;
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	// Commit to global memory
	*bccb_item = bccb_v;
}
