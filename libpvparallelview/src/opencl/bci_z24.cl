
/* Principle
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

	const float4 k = (float4)(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);

	float3 dummy;
	const float3 p = fabs(fract(c.xxx + k.xyz, &dummy) * 6.0f - k.www);

	const float3 r = clamp(p - k.xxx, value0, value1);

	return 0xFF000000 | (uint)(0xFF * r.x) << 16 | (uint)(0xFF * r.y) << 8 | (uint)(0xFF * r.z);
}

kernel void DRAW(const global uint2* bci_codes,
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

	for (int idx_y = get_local_id(1); idx_y < image_height; idx_y += get_local_size(1)) {
		shared_img[get_local_id(0) + idx_y*get_local_size(0)] = 0xFFFFFFFF;
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

		if (type == 0) {
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
		
		size_t idx = get_local_id(0) + pixel_y00*get_local_size(0);
		if (shared_img[idx] > shared_v) {
			shared_img[idx] = shared_v;
		}

		for (int pixel_y0 = pixel_y00+1; pixel_y0 < pixel_y01; pixel_y0++) {
			idx = get_local_id(0) + pixel_y0*get_local_size(0);
			if (shared_img[idx] > shared_v) {
				shared_img[idx] = shared_v;
			}
		}
	}

	band_x += image_x_start;

	if (reverse) {
		band_x = image_width-band_x-1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int idx_y = get_local_id(1); idx_y < image_height; idx_y += get_local_size(1)) {
		const uint pixel_shared = shared_img[get_local_id(0) + idx_y*get_local_size(0)];
		uint pixel;

		if (pixel_shared != 0xFFFFFFFF) {
			pixel = hue2rgb(pixel_shared & 0x000000FF);
		} else {
			pixel = 0x00000000;
		}
		image[band_x + idx_y*image_width] = pixel;
	}
}
