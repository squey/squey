// #include <apr_general.h>
// #include <apr_strings.h>
// #include <apr_tables.h>

#include <math.h>

#include <Qt/qcolor.h>
#include <QHash>
#include <QString>


#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/line-properties.h>
#include <picviz/utils.h>

#include <picviz/filtering-function.h>

#include <pvkernel/core/general.h>
#include <pvkernel/rush/pvformat.h>


/******************************************************************************
 *
 * picviz_filter_plugin_log_heatline_get_key
 *
 *****************************************************************************/
static QString picviz_filter_plugin_log_heatline_get_key(picviz_view_t *view, QStringList nrawvalues)
{
	QString key("");
	picviz_source_t *source = picviz_view_get_source_parent(view);

	for (int column = 0; column < nrawvalues.size(); column++) {
	        QString value = nrawvalues.at(column);

		if ( source->format->axes[column]["key"].compare("true", Qt::CaseInsensitive) == 0 ) {
		  key += QString(value);
		}
	}

	return key;
}


/******************************************************************************
 *
 * picviz_filter_plugin_log_heatline_colorize_do
 *
 *****************************************************************************/
void picviz_filter_plugin_log_heatline_colorize_do(picviz_view_t * /*view*/, picviz_layer_t *  /*input_layer*/, picviz_layer_t *output_layer, float ratio, PVRow line_id, float /*fmin*/, float /*fmax*/)
{

	Picviz::Color color;
	QColor qcolor;

	qcolor.setHsvF((1.0 - ratio)/3.0, 1.0, 1.0);
	color.fromQColor(qcolor);

	picviz_lines_properties_line_set_rgb_from_color(output_layer->lines_properties, line_id, color);
}




/******************************************************************************
 *
 * picviz_filter_plugin_log_heatline_select_do
 *
 *****************************************************************************/
void picviz_filter_plugin_log_heatline_select_do(picviz_view_t * /*view*/, picviz_layer_t * /*input_layer*/, picviz_layer_t *output_layer, float ratio, PVRow line_id, float fmin, float fmax) {

	if ((ratio > fmax) || (ratio < fmin)) {
		picviz_selection_set_line(output_layer->selection, line_id, 0);
	}
}



/******************************************************************************
 *
 * picviz_filter_plugin_log_heatline
 *
 *****************************************************************************/
void *picviz_filter_plugin_log_heatline(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_filtering_heatline_function function, float fmin, float fmax)
{
	PVRow nb_lines;
	PVRow counter;
	PVRow count_frequency;
	PVRow count_d;
	float count_f;
	PVRow highest_frequency;
	float highest_freq_f;
	//int columns;

	QHash<QString, PVRow> lines_hash;
	// apr_hash_t *lines_hash;

	picviz_nraw_t *nraw;

	QString key;

	float ratio;


	highest_frequency = 1;

	// lines_hash = apr_hash_make(view->pool);

	picviz_selection_A2B_copy(input_layer->selection, output_layer->selection);
	
	nb_lines = picviz_array_get_elts_count(view->parent->table);
	/* 1st round: we calculate all the frequencies */
	for (counter = 0; counter < nb_lines; counter++) {
		// apr_array_header_t *nrawvalues;
		QStringList nrawvalues;

		nraw = picviz_view_get_nraw_parent(view);

		nrawvalues = nraw->qt_nraw[counter];
		key = picviz_filter_plugin_log_heatline_get_key(view, nrawvalues);

		count_frequency = lines_hash[key];
		if (!count_frequency) {
		        lines_hash[key] = 1;
 		} else {
		        count_frequency++;
			if (count_frequency > highest_frequency) {
			  highest_frequency = count_frequency;
			}
			lines_hash[key] = count_frequency;
		}
	}

	/* 2nd round: we get the color from the ratio compared with the key and the frequency */
	for (counter = 0; counter < nb_lines; counter++) {
	        QStringList nrawvalues;
		nraw = picviz_view_get_nraw_parent(view);

		nrawvalues = nraw->qt_nraw[counter];
		key = picviz_filter_plugin_log_heatline_get_key(view, nrawvalues);

		count_frequency = lines_hash[key];

		/* ratio = (float)count_d / highest_frequency; */
		count_f = (float)count_frequency;
		highest_freq_f = (float)highest_frequency;

		ratio = logf(count_f) / logf(highest_freq_f);

		function(view, input_layer, output_layer, ratio, (PVRow)counter, fmin, fmax);
	}

// 	return selection;
	return NULL;
}

