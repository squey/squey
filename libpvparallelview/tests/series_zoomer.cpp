/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVSeriesViewZoomer.h>
#include <pvparallelview/common.h>

#include "common.h"

int main()
{
	PVViewZoomer vz;
	vz.resize(1000, 1000);
	PV_VALID(vz.currentZoom(), Zoom{0., 1., 0., 1.});
	vz.zoomIn(QRect{400, 0, 200, 1000});
	PV_VALID(vz.currentZoom(), Zoom{0.4, 0.6, 0., 1.});
}