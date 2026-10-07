
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
 * Squey has its hue starting at blue while standard HSV model starts with red:
 * squey: B C G Y R M B
 * standard : R Y G C B M R
 *
 * H_c = (N_color + R_i - H_i) mod N_color
 * where:
 * - H_c is the correct hue value
 * - H_i is the Squey hue value
 * - N_color is the number of color (see HSV_COLOR_COUNT)
 * - R_i is the index of the red color in Squey
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

/* What a pixel keeps the lowest of, among the lines crossing it, for a line of
 * an opacity of its own; the colour takes the 8 lower bits.
 *
 * The most opaque line wins, then the line of the lowest row, on the 15 upper
 * bits of its index, and black lines -- the zombies -- stay behind all the
 * others.
 */
uint translucent_pixel_value(const uint row, const uint color, const uint opacity)
{
	const uint black = color == HSV_COLOR_BLACK;

	return black << 31 | (255 - opacity) << 23 | (row >> 17) << 8 | color;
}

/* The line of the lowest row wins, on the 24 upper bits of its index, and black
 * lines stay behind all the others. Drawn by density, a code carries its opacity
 * in the 8 lower bits of its index instead, and lines are ordered as
 * translucent_pixel_value orders them. PVBCIDrawingBackendQPainter orders its
 * lines the same way.
 */
uint pixel_value(const uint row, const uint color, const uint density)
{
	if (density) {
		return translucent_pixel_value(row, color, row & 0xFF);
	}

	if (color == HSV_COLOR_BLACK) {
		return 0xFFFFFF00 | color;
	}

	return (row & 0xFFFFFF00) | color;
}

/* Antialiased, a line covers each pixel of a column by how close it passes to
 * its centre along the line's minor axis: vertically for a line flatter than a
 * diagonal, horizontally for a steeper one, as in Xiaolin Wu's algorithm. The
 * coverage scales the opacity of the line, which decides what the pixel keeps
 * (see translucent_pixel_value): where lines meet, the one covering the most of
 * the pixel is drawn, without blending with the others.
 *
 * y is the row the line passes at in the middle of the column, rows being
 * centred on whole values, and slope the rows it moves by per column. column
 * points to the first pixel of the column, pitch pixels apart from one row to
 * the next.
 */
void draw_covered(local uint* column,
                  const uint pitch,
                  const uint image_height,
                  const float y,
                  const float slope,
                  const uint row,
                  const uint color,
                  const uint opacity)
{
	const float reach = fmax(1.0f, fabs(slope));
	const float inverse_reach = 1.0f / reach;
	const int first = max((int)ceil(y - reach), 0);
	const int last = min((int)floor(y + reach), (int)image_height - 1);

	for (int pixel_y = first; pixel_y <= last; pixel_y++) {
		const float coverage = 1.0f - fabs((float)pixel_y - y) * inverse_reach;
		const uint alpha = (uint)(opacity * coverage + 0.5f);

		if (alpha == 0) {
			continue;
		}

		const uint value = translucent_pixel_value(row, color, alpha);
		local uint* const pixel = column + pixel_y * pitch;
		if (*pixel > value) {
			atomic_min(pixel, value);
		}
	}
}

//! The image is a QImage::Format_ARGB32_Premultiplied one.
uint premultiplied(const uint rgb, const uint opacity)
{
	const uint r = (((rgb >> 16) & 0xFF) * opacity + 127) / 255;
	const uint g = (((rgb >> 8) & 0xFF) * opacity + 127) / 255;
	const uint b = ((rgb & 0xFF) * opacity + 127) / 255;

	return opacity << 24 | r << 16 | g << 8 | b;
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
                 const uint reverse,
                 const uint density,
                 const uint antialiased)
{
	local uint shared_img[LOCAL_MEMORY_SIZE / sizeof(uint)];

	int band_x = get_local_id(0) + get_group_id(0)*get_local_size(0);

	/* The kernel is enqueued over a whole number of work-groups, so the last one
	 * may reach past the zone. Those work-items draw nothing, but they cannot
	 * leave: every work-item of a work-group has to reach the barriers below,
	 * and one returning early makes them undefined. Each loop they would take
	 * part in is guarded instead.
	 */
	const bool draws = band_x < width;

	const float alpha0 = (float)(width-band_x)/(float)width;
	const float alpha1 = (float)(width-(band_x+1))/(float)width;
	const float alpha_middle = ((float)(width-band_x) - 0.5f)/(float)width;
	const uint y_start = get_local_id(1) + get_group_id(1)*get_local_size(1);
	const uint y_pitch = get_local_size(1)*get_num_groups(1);

	for (int idx_y = get_local_id(1); draws && idx_y < image_height; idx_y += get_local_size(1)) {
		shared_img[get_local_id(0) + idx_y*get_local_size(0)] = 0xFFFFFFFF;
	}

	int pixel_y00;
	int pixel_y01;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (uint idx_codes = y_start; draws && idx_codes < n; idx_codes += y_pitch) {
		const uint2 code0 = bci_codes[idx_codes];

		const float l0 = (float) (code0.y & bit_mask);
		const int r0i = (code0.y >> bit_shift) & bit_mask;
		const int type = (code0.y >> ((2*bit_shift) + 8)) & 3;

		if (antialiased) {
			float y;
			float slope;

			if (type == 0) {
				const float r0 = (float) r0i;

				y = (r0 + ((l0-r0)*alpha_middle)) * zoom_y;
				slope = (r0-l0) * zoom_y / (float)width;
			} else {
				if (band_x > r0i) {
					continue;
				}

				// A line leaving the image within its first column keeps a finite slope.
				const float r0 = fmax((float) r0i, 1.0f);
				const float alpha_x = (type == 1 ? -l0 : (float)bit_mask-l0) / r0;

				y = (l0 + (alpha_x*((float)band_x + 0.5f))) * zoom_y;
				slope = alpha_x * zoom_y;
			}

			draw_covered(shared_img + get_local_id(0), get_local_size(0), image_height, y, slope,
			             code0.x, (code0.y >> 2*bit_shift) & 0xFF, density ? code0.x & 0xFF : 255);
			continue;
		}

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
		const uint shared_v = pixel_value(code0.x, color0, density);

		/* The work-items of a work-group that share an image column draw into the
		 * same pixels: the lowest value has to be kept atomically, or the line a
		 * pixel shows depends on which work-item happens to write last. Values only
		 * ever decrease, so reading first spares the atomic operation whenever the
		 * pixel already holds a lower one.
		 */
		size_t idx = get_local_id(0) + pixel_y00*get_local_size(0);
		if (shared_img[idx] > shared_v) {
			atomic_min(&shared_img[idx], shared_v);
		}

		for (int pixel_y0 = pixel_y00+1; pixel_y0 < pixel_y01; pixel_y0++) {
			idx = get_local_id(0) + pixel_y0*get_local_size(0);
			if (shared_img[idx] > shared_v) {
				atomic_min(&shared_img[idx], shared_v);
			}
		}
	}

	band_x += image_x_start;

	if (reverse) {
		band_x = image_width-band_x-1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int idx_y = get_local_id(1); draws && idx_y < image_height; idx_y += get_local_size(1)) {
		const uint pixel_shared = shared_img[get_local_id(0) + idx_y*get_local_size(0)];
		uint pixel;

		if (pixel_shared != 0xFFFFFFFF) {
			pixel = hue2rgb(pixel_shared & 0x000000FF);
			if (density || antialiased) {
				pixel = premultiplied(pixel, 255 - ((pixel_shared >> 23) & 0xFF));
			}
		} else {
			pixel = 0x00000000;
		}
		image[band_x + idx_y*image_width] = pixel;
	}
}
