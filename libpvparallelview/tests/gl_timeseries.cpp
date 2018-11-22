/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#include <inendi/PVRangeSubSampler.h>

#include <inendi/PVLayerFilter.h>

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>

#include <pvcop/core/selected_array.h>
#include <pvcop/core/range.h>

#include <limits>
#include <random>

#include "common.h"
#include <omp.h>

#include <pvkernel/rush/PVNraw.h>
#include <QApplication>

#include "PVSeriesView.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>

// datetime
#include <pvcop/types/datetime_us.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <pvcop/types/factory.h>

#include <pvlogger.h>

static constexpr const size_t ts_count = 40000;
static constexpr const size_t points_count = 2000;

static constexpr const char* csv_file = "/srv/logs/FUSION_MAINTENANCE_2016_cat_date.csv";
static constexpr const char* csv_file_format =
    "/srv/logs/FUSION_MAINTENANCE_2016_cat_date.csv.format";

int main(int argc, char** argv)
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::Source,
	                    "/srv/logs/constellium_nraw_small");

	env.compute_mappings();
	env.compute_plottings();
	env.compute_views();

	auto plotteds = env.root.get_children<Inendi::PVPlotted>();

	const auto& plotteds_vector = plotteds.front()->get_plotteds();

	Inendi::PVView* view = env.root.get_children<Inendi::PVView>().front();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	QApplication a(argc, argv);

	// a.setAttribute(Qt::AA_DontCreateNativeWidgetSiblings);

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (size_t i = 1; i < plotteds_vector.size(); i++) {
		// for (size_t i : { 18, 19, 20, 21 }) {
		timeseries.emplace_back(plotteds_vector[i].to_core_array<uint32_t>());
	}
	Inendi::PVRangeSubSampler sampler(nraw.column(PVCol(0)), timeseries);
	sampler.subsample(0, 200);

	PVSeriesView* myView = new PVSeriesView(sampler);

	myView->setBackgroundColor(QColor(10, 10, 10, 255));

	// myView->showMaximized();

	QGraphicsScene scene;
	scene.setBackgroundBrush(Qt::NoBrush);
	scene.setForegroundBrush(Qt::NoBrush);
	PVDecoratedSeriesView* dsv = new PVDecoratedSeriesView(&scene);
	// myView->resize(400,400);
	// dsv->setStyleSheet("background: transparent; border:none;");

	QGraphicsItem* item = scene.addText("QGraphicsTextItem");
	item->setFlags(QGraphicsItem::ItemIsMovable);

	// dsv->setRenderHint(QPainter::Antialiasing, false);
	// dsv->setOptimizationFlags(QGraphicsView::DontSavePainterState);
	dsv->setCacheMode(QGraphicsView::CacheBackground);
	dsv->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	dsv->setViewport(myView);

	dsv->resize(400, 400);

	// dsv->setParent(myView);
	// myView->showMaximized();

	dsv->show();
	return a.exec();
}
