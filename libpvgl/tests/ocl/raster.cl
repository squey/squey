__kernel void raster(__global const float* yl, __global const float* yr, const unsigned int nlines, const unsigned int line_offset, const float zoom_x, const float zoom_y, __global unsigned int* g_idxes, __local unsigned int* l_idxes)
{
	const unsigned int lidx = get_local_id(0);
	const unsigned int lidy = get_local_id(1);
	const unsigned int pixel_x = get_group_id(0)*get_local_size(0)+lidx;
	const unsigned int pixel_y = get_group_id(1)*get_local_size(1)+lidy;

	// Initialize local memory to 0
	__local unsigned int* item_id = &l_idxes[lidy*32 + lidx];
	__global unsigned int* gitem_id = &g_idxes[pixel_y*2048 + pixel_x];
	*item_id = *gitem_id;

	barrier(CLK_LOCAL_MEM_FENCE);

	bool found = false;
	if (*item_id != 0xFFFFFFFF) {
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}

	// This pixel is not "marked".
	// Compute for this chunk of lines
	float Yl,Yr,ytest;
	unsigned int Ytest;
	for (unsigned int i = 0; i < nlines; i++) {
		Yl = yl[line_offset+i]*zoom_x;
		Yr = yr[line_offset+i]*zoom_y;

		ytest = ((Yr-Yl)/zoom_x)*pixel_x+Yl;
		Ytest = (unsigned int) ytest;
		if (Ytest == pixel_y) {
			*item_id = line_offset + i;
			break;
		
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Copy local indexes back to global indexes
	*gitem_id = *item_id;
}
