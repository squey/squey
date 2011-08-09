//! \file PVAxes.cpp
//! $Id: PVAxes.cpp 3082 2011-06-08 07:57:47Z aguinet $
//! Copyright (C) Sebastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>

#include <QList>

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <picviz/general.h>
#include <picviz/PVView.h>
#include <picviz/PVMapping.h>

#include <pvgl/PVUtils.h>
#include <pvgl/views/PVParallel.h>
#include <pvgl/PVConfig.h>
#include <pvgl/PVWTK.h>

#include <pvgl/PVAxes.h>

/******************************************************************************
 *
 * PVGL::PVAxes::PVAxes
 *
 *****************************************************************************/
PVGL::PVAxes::PVAxes(PVView *view_): view(view_)
{
	PVLOG_DEBUG("PVGL::PVAxes::%s\n", __FUNCTION__);

	show_limits = false;
	vbo_position = 0;
	vbo_color = 0;
	vao = 0;
	vao_bg = 0;
	vbo_bg_position = 0;
}

PVGL::PVAxes::~PVAxes()
{
	PVLOG_INFO("In PVAxes destructor\n");

	if (vao != 0) {
		glDeleteVertexArrays(1, &vao);
	}
	if (vbo_position != 0) {
		glDeleteBuffers(1, &vbo_position);
	}
	if (vbo_color != 0) {
		glDeleteBuffers(1, &vbo_color);
	}
	if (vao_bg != 0) {
		glDeleteVertexArrays(1, &vao_bg);
	}
	if (vbo_bg_position != 0) {
		glDeleteBuffers(1, &vbo_bg_position);
	}
}

/******************************************************************************
 *
 * PVGL::PVAxes::init
 *
 *****************************************************************************/
void PVGL::PVAxes::init(Picviz::PVView_p pv_view_)
{
	PVLOG_DEBUG("PVGL::PVAxes::%s\n", __FUNCTION__);

	pv_view = pv_view_;

	std::vector<std::string> attributes;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo_position);
	attributes.push_back("position");
	glEnableVertexAttribArray(0);
	glGenBuffers(1, &vbo_color);
	attributes.push_back("color");
	glEnableVertexAttribArray(1);
	program = read_shader("parallel/axis.vert", "", "parallel/axis.frag", "","","", attributes);

	// Now for the background stuff
	attributes.clear();
	glGenVertexArrays(1, &vao_bg);
	glBindVertexArray(vao_bg);
	glGenBuffers(1, &vbo_bg_position);
	attributes.push_back("position");
	glEnableVertexAttribArray(0);
	program_bg = read_shader("parallel/axis_bg.vert", "", "parallel/axis_bg.frag", "","","", attributes);

	glBindVertexArray(0);
	glUseProgram(0);
}

/******************************************************************************
 *
 * PVGL::PVAxes::draw
 *
 *****************************************************************************/
void PVGL::PVAxes::draw(bool axes_mode)
{
	GLfloat m[16];
	Picviz::PVStateMachine *state_machine = pv_view->state_machine;

	PVLOG_HEAVYDEBUG("PVGL::PVAxes::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent())
		return;
	/* We check the ANTIALIASING mode */
	if (state_machine->is_antialiased()) {
		/* We activate ANTIALISASING */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);
	} else {
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
	}

	/* We set the LineWidth !*/
	glLineWidth(1.5f);

	//  We setup the vao.
	glUseProgram(program);
	glUniform1f(get_uni_loc(program, "time"), PVGL::wtk_time_ms_elapsed_since_init());
	if (axes_mode) {
		glUniform1i(get_uni_loc(program, "axis_mode"), 1);
	} else {
		glUniform1i(get_uni_loc(program, "axis_mode"), 0);
	}
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glUniformMatrix4fv(get_uni_loc(program, "modelview"), 1, GL_FALSE, m);
	glBindVertexArray(vao);
	glDrawArrays(GL_LINES, 0, position_array.size());
	glBindVertexArray(0);
	glUseProgram(0); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVAxes::draw_bg
 *
 *****************************************************************************/
void PVGL::PVAxes::draw_bg(void)
{
	GLfloat m[16];
	Picviz::PVStateMachine *state_machine = pv_view->state_machine;

	PVLOG_HEAVYDEBUG("PVGL::PVAxes::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent())
		return;
	if (!state_machine->is_axes_mode()) {
		return;
	}
	update_arrays_bg();

	/* We check the ANTIALIASING mode */
	if (state_machine->is_antialiased()) {
		/* We activate ANTIALISASING */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);
	} else {
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
	}

	// Do the real drawing.
	glDisable(GL_DEPTH_TEST);
	glUseProgram(program_bg);
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glUniformMatrix4fv(get_uni_loc(program_bg, "modelview"), 1, GL_FALSE, m);
	glBindVertexArray(vao_bg);
	glDrawArrays(GL_QUADS, 0, bg_position_array.size());
	glBindVertexArray(0);
	glUseProgram(0);
	glEnable(GL_DEPTH_TEST);
}

/******************************************************************************
 *
 * PVGL::PVAxes::draw_names
 *
 *****************************************************************************/
void PVGL::PVAxes::draw_names()
{
	int                  nb_axes;
	const float         *abscissae_list;
	float                font_size;
	Picviz::PVStateMachine *state_machine;

	PVLOG_HEAVYDEBUG("PVGL::PVAxes::%s\n", __FUNCTION__);

	if (!pv_view) { // Sanity check
		return;
	}
	if (!pv_view->is_consistent())
		return;
	state_machine  = pv_view->state_machine;
	nb_axes        = pv_view->get_axes_count();
	abscissae_list = pv_view->axes_combination.get_abscissae_list();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glOrtho(0,view->get_width(), view->get_height(),0, -1,1);

	int MX = std::max (1, view->get_width());
	int MY = std::max (1, view->get_height());

#ifdef SCREENSHOT
	font_size = 1.8 * std::min(22, std::max(10, int(MX / 500.0 * 12 * pow(1.2, view->zoom_level_x))));
#else
	font_size = std::min(22, std::max(10, int(MX / 500.0 * 12 * pow(1.2, view->zoom_level_x))));
#endif

	for (int i = 0; i < nb_axes; i++) {
		float gl_coord_x, gl_coord_y;
		float viewport_coord_x, viewport_coord_y;
		gl_coord_x = abscissae_list[i];
		gl_coord_y = 1.05;

		viewport_coord_x = MX*(0.5 + pow(1.2, view->zoom_level_x)/(view->xmax-view->xmin)*(gl_coord_x + view->translation.x - 0.5*(view->xmin+view->xmax)));
		viewport_coord_y = MY*(0.5 + pow(1.2, view->zoom_level_y)/(view->ymin-view->ymax)*(gl_coord_y + view->translation.y - 0.5*(view->ymin+view->ymax)));
		glPushMatrix();
		glTranslatef (viewport_coord_x, viewport_coord_y, 0);
		glRotatef (-45, 0, 0, 1);
		glTranslatef (-viewport_coord_x, -viewport_coord_y, 0);

	if (state_machine->is_axes_mode() && i == abscissae_list[pv_view->active_axis]) {
			glColor4ub(255, 0, 255, 255);
		} else {
			glColor4ubv(&pv_view->axes_combination.get_axis(i).get_titlecolor().x);
		}
		view->get_widget_manager().draw_text(viewport_coord_x, viewport_coord_y, qPrintable(pv_view->get_axis_name(i)), font_size);
		glPopMatrix();
	}

	if (show_limits) {
		Picviz::PVMapping_p mapping = pv_view->get_mapped_parent()->mapping;

		for (int i = 0; i < nb_axes; i++) {
			float gl_coord_x, gl_coord_y_min, gl_coord_y_max;
			float viewport_coord_x, viewport_coord_y_min, viewport_coord_y_max;
			QByteArray ymin;
			QByteArray ymax;
			{
				PVCol cur_axis = pv_view->axes_combination.get_axis_column_index(i);
				Picviz::mandatory_param_map const& mand_params = mapping->get_mandatory_params_for_col(cur_axis);
				Picviz::mandatory_param_map::const_iterator it_min = mand_params.find(Picviz::mandatory_ymin);
				Picviz::mandatory_param_map::const_iterator it_max = mand_params.find(Picviz::mandatory_ymax);
				if (it_min == mand_params.end() || it_max == mand_params.end()) {
					PVLOG_WARN("ymin and/or ymax don't exist for axis %d. Maybe the mandatory minmax mapping hasn't be run ?\n", cur_axis);
					continue;
				}
				ymin = (*it_min).second.first.toLocal8Bit();
				ymax = (*it_max).second.first.toLocal8Bit();
			}
			gl_coord_x = abscissae_list[i];
			gl_coord_y_max = 1;
			gl_coord_y_min = 0;

			viewport_coord_x = MX*(0.5 + pow(1.2, view->zoom_level_x)/(view->xmax-view->xmin)*(gl_coord_x + view->translation.x - 0.5*(view->xmin+view->xmax))) + 3;
			viewport_coord_y_min = MY*(0.5 + pow(1.2, view->zoom_level_y)/(view->ymin-view->ymax)*(gl_coord_y_min + view->translation.y - 0.5*(view->ymin+view->ymax)));
			viewport_coord_y_max = MY*(0.5 + pow(1.2, view->zoom_level_y)/(view->ymin-view->ymax)*(gl_coord_y_max + view->translation.y - 0.5*(view->ymin+view->ymax)));
			glColor4ubv(&pv_view->axes_combination.get_axis(i).get_titlecolor().x);
			glPushMatrix();
			glTranslatef (viewport_coord_x, viewport_coord_y_min, 0);
			glRotatef (45, 0, 0, 1);
			glTranslatef (-viewport_coord_x, -viewport_coord_y_min, 0);
			view->get_widget_manager().draw_text(viewport_coord_x, viewport_coord_y_min, ymin.data(), font_size / 2.0);
			glPopMatrix();
			glPushMatrix();
			glTranslatef (viewport_coord_x, viewport_coord_y_max, 0);
			glRotatef (-45, 0, 0, 1);
			glTranslatef (-viewport_coord_x, -viewport_coord_y_max, 0);
			view->get_widget_manager().draw_text(viewport_coord_x, viewport_coord_y_max, ymax.data(), font_size/ 2.0);
			glPopMatrix();
		}
	}
}

/******************************************************************************
 *
 * PVGL::PVAxes::update_arrays
 *
 *****************************************************************************/
void PVGL::PVAxes::update_arrays (void)
{
	const float *abscissae_list = pv_view->axes_combination.get_abscissae_list();
	//picviz_axis_t **axes_list = pv_view->axes_combination->axes_list;
	int nb_axes = pv_view->get_axes_count();

	PVLOG_DEBUG("PVGL::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent())
		return;
	position_array.clear();
	color_array.clear();
	for (int i = 0; i < nb_axes; i++) {
		position_array.push_back(vec3(abscissae_list[i], 0.0, 300.0));
		position_array.push_back(vec3(abscissae_list[i], 1.0, 300.0));
		color_array.push_back(ubvec4(&pv_view->axes_combination.get_axis(i).get_color().x));
		color_array.push_back(ubvec4(&pv_view->axes_combination.get_axis(i).get_color().x));

/*		std::cout << "---------------------" << std::endl;
		std::cout << "id: "           << axes_list[j]->id << std::endl;
		std::cout << "absciss: "      << axes_list[j]->absciss << std::endl;
		std::cout << "column_index: " << axes_list[j]->column_index << std::endl;
		std::cout << "color: "        << int(axes_list[j]->color->r) << ", " << int(axes_list[j]->color->g) << ", " << int(axes_list[j]->color->b) << std::endl;
		std::cout << "flags: "        << (axes_list[j]->is_expandable ? "expandable ":"") <<
				                             (axes_list[j]->is_expanded ? "expanded ":"") <<
																		 (axes_list[j]->is_key ? "key":"") << std::endl;
		std::cout << "name: "         << axes_list[j]->name << std::endl;
//		std::cout << "type: "         << axes_list[j]->type << std::endl;
//		std::cout << "modemapping: "  << axes_list[j]->modemapping << std::endl;
//		std::cout << "modeplotting: " << axes_list[j]->modeplotting << std::endl;
		std::cout << "thickness: "    << axes_list[j]->thickness << std::endl;*/
	}
	glBindVertexArray(vao);
	glBindBuffer (GL_ARRAY_BUFFER, vbo_position);
	glBufferData (GL_ARRAY_BUFFER, position_array.size() * sizeof (vec3), &position_array[0], GL_STATIC_DRAW);
	glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer (GL_ARRAY_BUFFER, vbo_color);
	glBufferData (GL_ARRAY_BUFFER, color_array.size() * sizeof (ubvec4), &color_array[0], GL_STATIC_DRAW);
	glVertexAttribPointer (1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
	glBindVertexArray(0);
}

/******************************************************************************
 *
 * PVGL::PVAxes::update_arrays_bg
 *
 *****************************************************************************/
void PVGL::PVAxes::update_arrays_bg (void)
{
	Picviz::PVStateMachine *state_machine = pv_view->state_machine;

	PVLOG_DEBUG("PVGL::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent())
		return;
	if (state_machine->is_axes_mode()) {
		const float *abscissae_list = pv_view->axes_combination.get_abscissae_list();
		float  x = abscissae_list[pv_view->active_axis];

		bg_position_array.clear();
		glBindVertexArray(vao_bg);
		bg_position_array.clear();
		bg_position_array.push_back(vec3(x - 0.1, -0.025, 0.0));
		bg_position_array.push_back(vec3(x - 0.1, 1.025, 0.0));
		bg_position_array.push_back(vec3(x + 0.1, 1.025, 0.0));
		bg_position_array.push_back(vec3(x + 0.1, -0.025, 0.0));
		glEnableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_bg_position);
		glBufferData(GL_ARRAY_BUFFER, bg_position_array.size() * sizeof (vec3), &bg_position_array[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glBindVertexArray(0);
	}
}

/******************************************************************************
 *
 * PVGL::PVAxes::toggle_show_limits
 *
 *****************************************************************************/
void PVGL::PVAxes::toggle_show_limits()
{
	show_limits = !show_limits;
}
