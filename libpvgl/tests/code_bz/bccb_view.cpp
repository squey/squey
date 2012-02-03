#include <common/common.h>
#include <code_bz/types.h>
#include <gl/bccb_view.h>

#include <tbb/tick_count.h>

void BCCBView::paintGL()
{
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage BZ" << std::endl;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);

	glBegin(GL_LINES);
	uint16_t lx,ly,rx,ry;
	for (size_t i = 0; i < NB_INT_BCODECB; i++) {
		uint32_t tmp = _cb[i];
		for (int b = 0; b < 32; b++) {
			if ((tmp & (1<<b)) != 0) {
				PVBCode code;
				code.int_v = (i<<5) + b;
				code.to_pts(_ortho_w, _ortho_h, lx, ly, rx, ry);
				glVertex2i(lx, ly);
				glVertex2i(rx, ry);
			}
		}
	}
	glEnd();

	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage CB BZ: " << (end-start).seconds() << std::endl;
}
