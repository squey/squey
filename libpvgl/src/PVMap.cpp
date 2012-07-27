/**
 * \file PVMap.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

#include <picviz/PVView.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVLines.h>
#include <pvgl/views/PVParallel.h>
#include <pvgl/PVMain.h>
#include <pvgl/PVWTK.h>

#include <pvgl/PVMap.h>


const int MAP_FBO_MAX_WIDTH = 2048;
const int MAP_FBO_MAX_HEIGHT= 1024;

/******************************************************************************
 *
 * PVGL::PVMap::PVMap
 *
 *****************************************************************************/
PVGL::PVMap::PVMap(PVView *view_, PVWidgetManager *widget_manager_, PVLines *lines_, int width, int height) :
view(view_), widget_manager(widget_manager_), lines(lines_)
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	allocation.width = width / 4;
	allocation.height= height / 4;
	allocation.x = PVGL_MAP_DEFAULT_X;
	allocation.y = PVGL_MAP_DEFAULT_Y;
	grabbing = false;
	dragging = false;
	move_view_mode = false;
	main_fbo = 0;
	main_fbo_vao = 0;
	main_fbo_vbo = 0;
	main_fbo_tex = 0;
	lines_fbo = 0;
	lines_fbo_tex = 0;
	zombie_fbo = 0;
	zombie_fbo_tex = 0;
}

/******************************************************************************
 *
 * PVGL::PVMap::~PVMap
 *
 *****************************************************************************/
PVGL::PVMap::~PVMap()
{
	free_buffers();
}

void PVGL::PVMap::free_buffers()
{
	if (main_fbo) {
		glDeleteFramebuffers(1, &main_fbo);
		main_fbo = 0;
	}
	if (main_fbo_tex) {
		glDeleteTextures(1, &main_fbo_tex);
		main_fbo_tex = 0;
	}
	if (main_fbo_vao) {
		glDeleteVertexArrays(1, &main_fbo_vao);
		main_fbo_vao = 0;
	}
	if (main_fbo_vbo) {
		glDeleteBuffers(1, &main_fbo_vbo);
		main_fbo_vbo = 0;
	}
	if (lines_fbo) {
		glDeleteFramebuffers(1, &lines_fbo);
		lines_fbo = 0;
	}
	if (lines_fbo_tex) {
		glDeleteTextures(1, &lines_fbo_tex);
		lines_fbo_tex = 0;
	}
	if (zombie_fbo) {
		glDeleteFramebuffers(1, &zombie_fbo);
		zombie_fbo = 0;
	}
	if (zombie_fbo_tex) {
		glDeleteTextures(1, &zombie_fbo_tex);
		zombie_fbo_tex = 0;
	}
}

/******************************************************************************
 *
 * PVGL::PVMap::init
 *
 *****************************************************************************/
void PVGL::PVMap::init(Picviz::PVView_p pv_view_)
{
	free_buffers();
	std::vector<std::string> attributes;

	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	picviz_view = pv_view_;
	// The Map's VAO
	glGenVertexArrays(1, &main_fbo_vao); PRINT_OPENGL_ERROR();
	glBindVertexArray(main_fbo_vao); PRINT_OPENGL_ERROR();
	glGenBuffers(1, &main_fbo_vbo); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, main_fbo_vbo);
	glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), 0, GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(0); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0); PRINT_OPENGL_ERROR();
	attributes.push_back("position");
	glEnableVertexAttribArray(1); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), BUFFER_OFFSET(2 * sizeof(GLfloat))); PRINT_OPENGL_ERROR();
	attributes.push_back("tex_coord"); PRINT_OPENGL_ERROR();
	main_fbo_program = read_shader("parallel/lines_map_fbo.vert", "", "parallel/lines_map_fbo.frag", "", "", "", attributes);
	glUniform1i(get_uni_loc(main_fbo_program, "map_fbo_sampler"), 0); PRINT_OPENGL_ERROR();

	// And a vao for the map mask.
	init_mask();

	// And the auxiliary fbos.
	init_fbo();
	init_lines_fbo();
	init_zombie_fbo();

	// Restore a sane state.
	glUseProgram(0); PRINT_OPENGL_ERROR();
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	visible = false;
}

/******************************************************************************
 *
 * PVGL::PVMap::init_fbo
 *
 *****************************************************************************/
void PVGL::PVMap::init_fbo()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	glGenFramebuffers(1, &main_fbo); PRINT_OPENGL_ERROR();
	glBindFramebuffer(GL_FRAMEBUFFER, main_fbo); PRINT_OPENGL_ERROR();
	glGenTextures(1, &main_fbo_tex); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_RECTANGLE, main_fbo_tex); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, MAP_FBO_MAX_WIDTH, MAP_FBO_MAX_HEIGHT, 0,
	             GL_RGBA, GL_UNSIGNED_BYTE, 0); PRINT_OPENGL_ERROR();
	glFramebufferTexture2D(GL_FRAMEBUFFER,
	                       GL_COLOR_ATTACHMENT0,
	                       GL_TEXTURE_RECTANGLE, main_fbo_tex, 0); PRINT_OPENGL_ERROR();
	check_framebuffer_status();
	glBindFramebuffer(GL_FRAMEBUFFER, 0); PRINT_OPENGL_ERROR();
	set_main_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::init_lines_fbo
 *
 *****************************************************************************/
void PVGL::PVMap::init_lines_fbo()
{
	GLuint depth_rb;

	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	glGenFramebuffers(1, &lines_fbo); PRINT_OPENGL_ERROR();
	glBindFramebuffer(GL_FRAMEBUFFER, lines_fbo); PRINT_OPENGL_ERROR();
	glGenTextures(1, &lines_fbo_tex); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_RECTANGLE, lines_fbo_tex); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, MAP_FBO_MAX_WIDTH, MAP_FBO_MAX_HEIGHT, 0,
	             GL_RGBA, GL_UNSIGNED_BYTE, 0); PRINT_OPENGL_ERROR();
	glFramebufferTexture2D(GL_FRAMEBUFFER,
	                       GL_COLOR_ATTACHMENT0,
	                       GL_TEXTURE_RECTANGLE, lines_fbo_tex, 0); PRINT_OPENGL_ERROR();
	glGenRenderbuffers(1, &depth_rb); PRINT_OPENGL_ERROR();
	glBindRenderbuffer(GL_RENDERBUFFER, depth_rb); PRINT_OPENGL_ERROR();
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, MAP_FBO_MAX_WIDTH, MAP_FBO_MAX_HEIGHT); PRINT_OPENGL_ERROR();
	glFramebufferRenderbuffer(GL_FRAMEBUFFER,
	                          GL_DEPTH_ATTACHMENT,
	                          GL_RENDERBUFFER,
	                          depth_rb); PRINT_OPENGL_ERROR();
	check_framebuffer_status();
	glBindFramebuffer(GL_FRAMEBUFFER, 0); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVMap::init_zombie_fbo()
 *
 *****************************************************************************/
void PVGL::PVMap::init_zombie_fbo()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	glGenFramebuffers(1, &zombie_fbo); PRINT_OPENGL_ERROR();
	glBindFramebuffer(GL_FRAMEBUFFER, zombie_fbo); PRINT_OPENGL_ERROR();
	glGenTextures(1, &zombie_fbo_tex); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_RECTANGLE, zombie_fbo_tex); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, MAP_FBO_MAX_WIDTH, MAP_FBO_MAX_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); PRINT_OPENGL_ERROR();
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, zombie_fbo_tex, 0); PRINT_OPENGL_ERROR();
	check_framebuffer_status();
	glBindFramebuffer(GL_FRAMEBUFFER, 0); PRINT_OPENGL_ERROR();

	set_zombie_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::init_mask()
 *
 *****************************************************************************/
void PVGL::PVMap::init_mask()
{
	std::vector<std::string> attributes;

	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	glGenVertexArrays(1, &mask_vao); PRINT_OPENGL_ERROR();
	glBindVertexArray(mask_vao); PRINT_OPENGL_ERROR();
	glGenBuffers(1, &mask_vbo); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, mask_vbo); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), 0, GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(0); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); PRINT_OPENGL_ERROR();
	attributes.push_back("position");
	mask_program = read_shader("parallel/lines_map_mask.vert", "", "parallel/lines_map_mask.frag", "", "", "", attributes);
}

/******************************************************************************
 *
 * PVGL::PVMap::set_main_fbo_dirty
 *
 *****************************************************************************/
void PVGL::PVMap::set_main_fbo_dirty()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	main_fbo_dirty = true;
}

/******************************************************************************
 *
 * PVGL::PVMap::set_lines_fbo_dirty
 *
 *****************************************************************************/
void PVGL::PVMap::set_lines_fbo_dirty()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);
	if (!visible) {
		return;
	}

	lines_fbo_dirty = true;
	drawn_lines = 0;
	idle_manager.new_task(view, IDLE_REDRAW_MAP_LINES);
	set_main_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::set_zombie_fbo_dirty
 *
 *****************************************************************************/
void PVGL::PVMap::set_zombie_fbo_dirty()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);
	if (!visible) {
		return;
	}

	zombie_fbo_dirty = true;
	drawn_zombie_lines = 0;
	idle_manager.new_task(view, IDLE_REDRAW_ZOMBIE_MAP_LINES);
	set_main_fbo_dirty();
}
/******************************************************************************
 *
 * PVGL::PVMap::update_arrays_z
 *
 *****************************************************************************/
void PVGL::PVMap::update_arrays_z()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	set_lines_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::update_arrays_colors
 *
 *****************************************************************************/
void PVGL::PVMap::update_arrays_colors()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	set_lines_fbo_dirty();
	set_zombie_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::update_arrays_selection
 *
 *****************************************************************************/
void PVGL::PVMap::update_arrays_selection()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	set_lines_fbo_dirty();

	//update_arrays_colors();
}

/******************************************************************************
 *
 * PVGL::PVMap::update_arrays_zombies
 *
 *****************************************************************************/
void PVGL::PVMap::update_arrays_zombies()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	set_zombie_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::update_arrays_positions
 *
 *****************************************************************************/
void PVGL::PVMap::update_arrays_positions()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	set_lines_fbo_dirty();
	set_zombie_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::set_size
 *
 *****************************************************************************/
void PVGL::PVMap::set_size(int width, int height)
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	allocation.width = width / 4;
	allocation.height= height/ 4;
	set_lines_fbo_dirty();
	set_zombie_fbo_dirty();
}

/******************************************************************************
 *
 * PVGL::PVMap::draw_zombie_lines
 *
 *****************************************************************************/
void PVGL::PVMap::draw_zombie_lines(GLfloat modelview[16])
{
	int nb_lines_to_draw = idle_manager.get_number_of_lines(view, IDLE_REDRAW_ZOMBIE_MAP_LINES);

	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, zombie_fbo); PRINT_OPENGL_ERROR();
	glViewport(0, 0, allocation.width, allocation.height); PRINT_OPENGL_ERROR();
	glClearColor(0.0, 0.0, 0.0, 0.0);
	if (drawn_zombie_lines == 0) {
		glClear(GL_COLOR_BUFFER_BIT); PRINT_OPENGL_ERROR();
	}
	if (nb_lines_to_draw == 0) {
		return;
	}
	glEnable(GL_BLEND);
	glBlendEquation(GL_MAX);
	for (unsigned i = 0; i < lines->nb_batches; i++) {
		glUseProgram(lines->batches[i].zombie_program); PRINT_OPENGL_ERROR();
		glUniform2f(get_uni_loc(lines->batches[i].zombie_program, "zoom"), 1.0, 1.0); PRINT_OPENGL_ERROR();
		glUniformMatrix4fv(get_uni_loc(lines->batches[i].zombie_program, "view"), 1, GL_FALSE, modelview); PRINT_OPENGL_ERROR ();
		glUniform1i(get_uni_loc(lines->batches[i].zombie_program, "drawn_lines"), drawn_zombie_lines); PRINT_OPENGL_ERROR();
		glBindVertexArray(lines->batches[i].vao); PRINT_OPENGL_ERROR();

		// FIXME: this is probably overkill to do these 3 lines here, but I do have curious error if I remove them.
		glActiveTexture(GL_TEXTURE2); PRINT_OPENGL_ERROR();
		glBindTexture(GL_TEXTURE_BUFFER, lines->tbo_zombie_texture); PRINT_OPENGL_ERROR();
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, lines->tbo_zombie); PRINT_OPENGL_ERROR();
		
		glDrawArrays(GL_POINTS, drawn_zombie_lines, picviz_min(nb_lines_to_draw, int(picviz_view->get_row_count() - drawn_zombie_lines)));
	}
	glBlendEquation(GL_FUNC_ADD);
	glDisable(GL_BLEND);
	drawn_zombie_lines += nb_lines_to_draw;

	if (drawn_zombie_lines >= int(picviz_view->get_row_count())) {
		idle_manager.remove_task(view, IDLE_REDRAW_ZOMBIE_MAP_LINES);
		zombie_fbo_dirty = false;
		drawn_zombie_lines = 0;
	}
}

/******************************************************************************
 *
 * PVGL::PVMap::draw_selected_lines
 *
 *****************************************************************************/
void PVGL::PVMap::draw_selected_lines(GLfloat modelview[16])
{
	int nb_lines_to_draw = idle_manager.get_number_of_lines(view, IDLE_REDRAW_MAP_LINES);

	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	if (nb_lines_to_draw == 0) {
		return;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, lines_fbo); PRINT_OPENGL_ERROR();
	glViewport(0, 0, allocation.width, allocation.height); PRINT_OPENGL_ERROR();
	glClearColor(0.0, 0.0, 0.0, 0.0); PRINT_OPENGL_ERROR();
	if (drawn_lines == 0) {
		glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT); PRINT_OPENGL_ERROR();
	}
	for (unsigned i = 0; i < lines->nb_batches; i++) {
		glUseProgram(lines->batches[i].program); PRINT_OPENGL_ERROR();
		glUniform2f(get_uni_loc(lines->batches[i].program, "zoom"), 1.0, 1.0); PRINT_OPENGL_ERROR();
		glUniformMatrix4fv (get_uni_loc (lines->batches[i].program, "view"), 1, GL_FALSE, modelview); PRINT_OPENGL_ERROR ();
		glUniform1i(get_uni_loc(lines->batches[i].program, "eventline_first"), picviz_view->eventline.get_first_index()); PRINT_OPENGL_ERROR();
		glUniform1i(get_uni_loc(lines->batches[i].program, "eventline_current"), picviz_view->eventline.get_current_index()); PRINT_OPENGL_ERROR();
		glUniform1i(get_uni_loc(lines->batches[i].program, "drawn_lines"), drawn_lines); PRINT_OPENGL_ERROR();

		glBindVertexArray(lines->batches[i].vao); PRINT_OPENGL_ERROR();
		// FIXME: this is probably overkill to do these 3 lines here, but I do have curious error if I remove them.
		glActiveTexture(GL_TEXTURE1); PRINT_OPENGL_ERROR();
		glBindTexture(GL_TEXTURE_BUFFER, lines->tbo_selection_texture); PRINT_OPENGL_ERROR();
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, lines->tbo_selection); PRINT_OPENGL_ERROR();

		glDrawArrays(GL_POINTS, drawn_lines, picviz_min(nb_lines_to_draw, int( picviz_view->get_row_count() - drawn_lines)));
	}
	drawn_lines += nb_lines_to_draw;
	if (drawn_lines >= int(picviz_view->get_row_count())) {
		idle_manager.remove_task(view, IDLE_REDRAW_MAP_LINES);
		lines_fbo_dirty = false;
		drawn_lines = 0;
	}
}

/******************************************************************************
 *
 * PVGL::PVMap::draw
 *
 *****************************************************************************/
void PVGL::PVMap::draw()
{
	Picviz::PVStateMachine *state_machine = picviz_view->state_machine;

	PVLOG_HEAVYDEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	if (!visible)
		return;
	GLfloat modelview[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview); PRINT_OPENGL_ERROR();
	glLineWidth(1.0);

	if (main_fbo_dirty) { // We need to redraw the lines into the framebuffer.
		GLfloat modelview_map[16];

		// Setup matrices.
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glOrtho(-0.5f, picviz_view->get_axes_count() - 0.5f, -0.1f, 1.1f, -350.0f, 350.0f);
		glViewport(0, 0, allocation.width, allocation.height); PRINT_OPENGL_ERROR();
		glGetFloatv(GL_MODELVIEW_MATRIX, modelview_map); PRINT_OPENGL_ERROR();

		// Draw the zombie/non-zombie lines into their own FBO
		if (!state_machine->is_grabbed() && !is_panning()) {
			if (zombie_fbo_dirty && (state_machine->are_gl_zombie_visible() || state_machine->are_gl_unselected_visible())) {
				draw_zombie_lines(modelview_map);
			}
			// Draw the selected lines into their own FBO
			if (lines_fbo_dirty) {
				draw_selected_lines(modelview_map);
			}
		}
		glDisable(GL_DEPTH_TEST);

		// Draw into the main FBO
		glViewport(0, 0, allocation.width, allocation.height); PRINT_OPENGL_ERROR();
		glBindFramebuffer(GL_FRAMEBUFFER, main_fbo); PRINT_OPENGL_ERROR();

		// Draw the zombies fbo into the main FBO
			{
				glActiveTexture(GL_TEXTURE0); PRINT_OPENGL_ERROR();
				glEnable(GL_TEXTURE_RECTANGLE); PRINT_OPENGL_ERROR();
				glBindTexture(GL_TEXTURE_RECTANGLE, zombie_fbo_tex); PRINT_OPENGL_ERROR();
				glUseProgram(lines->zombie_fbo_program); PRINT_OPENGL_ERROR();
				glUniform2f(get_uni_loc(lines->zombie_fbo_program, "zoom"), 1, 1);
				glUniform2f(get_uni_loc(lines->zombie_fbo_program, "offset"), 0, 0); PRINT_OPENGL_ERROR();
				glUniform2f(get_uni_loc(lines->zombie_fbo_program, "size"), allocation.width, allocation.height); PRINT_OPENGL_ERROR();
				glUniform1i(get_uni_loc(lines->zombie_fbo_program, "draw_zombie"), state_machine->are_gl_zombie_visible()); PRINT_OPENGL_ERROR();
				glUniform1i(get_uni_loc(lines->zombie_fbo_program, "draw_unselected"), state_machine->are_gl_unselected_visible()); PRINT_OPENGL_ERROR();
				glBindVertexArray(lines->fbo_vao); PRINT_OPENGL_ERROR();
				glDrawArrays(GL_QUADS, 0, 4); PRINT_OPENGL_ERROR();
			}
		// Draw the "selected lines" fbo into the main FBO
			{
				glBindTexture(GL_TEXTURE_RECTANGLE, lines_fbo_tex); PRINT_OPENGL_ERROR();
				glUseProgram(lines->main_fbo_program); PRINT_OPENGL_ERROR();
				glUniform2f(get_uni_loc(lines->main_fbo_program, "zoom"), 1, 1);
				glUniform2f(get_uni_loc(lines->main_fbo_program, "offset"), 0, 0); PRINT_OPENGL_ERROR();
				glUniform2f(get_uni_loc(lines->main_fbo_program, "size"), allocation.width, allocation.height); PRINT_OPENGL_ERROR();
				glBindVertexArray(lines->fbo_vao); PRINT_OPENGL_ERROR();
				//glDisable(GL_TEXTURE_RECTANGLE); glUseProgram(0);glColor4f(1.0,0.0,1.0,0.5);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glDrawArrays(GL_QUADS, 0, 4); PRINT_OPENGL_ERROR();
				glDisable(GL_BLEND);
			}
		glColor3f(1.0f, 1.0f, 1.0f);
		glEnable(GL_DEPTH_TEST);

		glActiveTexture(GL_TEXTURE0); PRINT_OPENGL_ERROR();
		glUseProgram(0); PRINT_OPENGL_ERROR();
		glBindVertexArray(0); PRINT_OPENGL_ERROR();

		view->draw_axes();
		view->draw_selection_square();
		if (!lines_fbo_dirty && !zombie_fbo_dirty) {
			main_fbo_dirty = false;
		}
	}
	glViewport(0, 0, view->get_width(), view->get_height()); PRINT_OPENGL_ERROR();
	glBindFramebuffer(GL_FRAMEBUFFER, 0); PRINT_OPENGL_ERROR();
	glDisable(GL_BLEND);

	// Draw the map in the view.
	glDisable(GL_DEPTH_TEST); PRINT_OPENGL_ERROR();
	glDisable(GL_TEXTURE_RECTANGLE); PRINT_OPENGL_ERROR();
	glUseProgram(0);
	// Draw the gui (the window's border, that is).
	glLoadIdentity();
	glOrtho(0, view->get_width(), view->get_height(),0, -1,1);
	widget_manager->draw_icon(allocation.x,             view->get_height() - allocation.y - allocation.height, PVGL_ICON_WINDOW_LT);
	widget_manager->draw_icon_streched(allocation.x+26, view->get_height() - allocation.y - allocation.height, allocation.width-26-22,32, PVGL_ICON_WINDOW_T);
	widget_manager->draw_icon(allocation.x + allocation.width, view->get_height() - allocation.y - allocation.height, PVGL_ICON_WINDOW_RT);

	widget_manager->draw_icon_streched(allocation.x, view->get_height() - allocation.y - allocation.height+18, 32, allocation.height-18-21, PVGL_ICON_WINDOW_L);

	widget_manager->draw_icon(allocation.x,             view->get_height() - allocation.y,              PVGL_ICON_WINDOW_LB);
	widget_manager->draw_icon_streched(allocation.x+26, view->get_height() - allocation.y, allocation.width-26-22,32, PVGL_ICON_WINDOW_B);
	widget_manager->draw_icon(allocation.x + allocation.width, view->get_height() - allocation.y,              PVGL_ICON_WINDOW_RB);

	widget_manager->draw_icon_streched(allocation.x + allocation.width, view->get_height() - allocation.y - allocation.height+18, 32, allocation.height-18-21, PVGL_ICON_WINDOW_R);

	// And draw the main fbo.
	glBindTexture(GL_TEXTURE_RECTANGLE, main_fbo_tex); PRINT_OPENGL_ERROR();
	glUseProgram(main_fbo_program); PRINT_OPENGL_ERROR();
	glUniform2f(get_uni_loc(main_fbo_program, "size_map"), allocation.width, allocation.height); PRINT_OPENGL_ERROR();
	glUniform2f(get_uni_loc(main_fbo_program, "size"), view->get_width(), view->get_height()); PRINT_OPENGL_ERROR();
	glBindVertexArray(main_fbo_vao); PRINT_OPENGL_ERROR();
	GLfloat fbo_vertex_buffer[] = {
		(GLfloat) allocation.x,                    (GLfloat) allocation.y,                     0.0f, 0.0f,
		(GLfloat) (allocation.x + allocation.width), (GLfloat) allocation.y,                     1.0f, 0.0f,
		(GLfloat) (allocation.x + allocation.width), (GLfloat) (allocation.y + allocation.height), 1.0f, 1.0f,
		(GLfloat) allocation.x,                    (GLfloat) (allocation.y + allocation.height), 0.0f, 1.0f
	};
	glBindBuffer(GL_ARRAY_BUFFER, main_fbo_vbo); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, sizeof fbo_vertex_buffer,
	             fbo_vertex_buffer, GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
	glDrawArrays(GL_QUADS, 0, 4); PRINT_OPENGL_ERROR();

	// Draw the mask.
		{
			GLfloat modelview_map[16];
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glOrtho(-0.5f, picviz_view->get_axes_count() - 0.5f, -0.1f, 1.1f, -350.0f, 350.0f);
			glViewport(allocation.x, allocation.y, allocation.width, allocation.height); PRINT_OPENGL_ERROR();
			glGetFloatv(GL_MODELVIEW_MATRIX, modelview_map); PRINT_OPENGL_ERROR();
			glBindVertexArray(mask_vao); PRINT_OPENGL_ERROR();
			glBindBuffer(GL_ARRAY_BUFFER, mask_vbo); PRINT_OPENGL_ERROR();
			float min_x = -0.5f;
			float min_y = -0.1f;
			float max_x = picviz_view->get_axes_count() - 0.5f;
			float max_y = 1.1f;
			GLfloat mask_position[] = {
				min_x, min_y,
				min_x, max_y,
				max_x, max_y,
				max_x, min_y
			};
			glBufferData(GL_ARRAY_BUFFER, sizeof mask_position, mask_position, GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
			glUseProgram(mask_program); PRINT_OPENGL_ERROR();
			glDisable(GL_DEPTH_TEST); PRINT_OPENGL_ERROR();
			vec2 selection_min, selection_max;
			selection_min.x = (view->xmin + view->xmax) / 2.0 + (view->xmax - view->xmin) * (0.0 - 0.5) / pow(1.2, view->zoom_level_x) - view->translation.x;
			selection_min.y = (view->ymin + view->ymax) / 2.0 + (view->ymin - view->ymax) * (0.0 - 0.5) / pow(1.2, view->zoom_level_y) - view->translation.y;
			selection_max.x = (view->xmin + view->xmax) / 2.0 + (view->xmax - view->xmin) * (1.0 - 0.5) / pow(1.2, view->zoom_level_x) - view->translation.x;
			selection_max.y = (view->ymin + view->ymax) / 2.0 + (view->ymin - view->ymax) * (1.0 - 0.5) / pow(1.2, view->zoom_level_y) - view->translation.y;

			glUniformMatrix4fv (get_uni_loc (mask_program, "view"), 1, GL_FALSE, modelview_map); PRINT_OPENGL_ERROR ();
			glUniform2f(get_uni_loc(mask_program, "min_mask"), selection_min.x, selection_max.y);
			glUniform2f(get_uni_loc(mask_program, "max_mask"), selection_max.x, selection_min.y);
			glEnable(GL_BLEND); PRINT_OPENGL_ERROR();
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); PRINT_OPENGL_ERROR();
			glDrawArrays(GL_QUADS, 0, 4);
			glViewport(0, 0, view->get_width(), view->get_height()); PRINT_OPENGL_ERROR();
		}

	// Restore a known state
	glDisable(GL_TEXTURE_RECTANGLE); PRINT_OPENGL_ERROR();
	glUseProgram(0); PRINT_OPENGL_ERROR();
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	glLoadMatrixf(modelview);

	glEnable(GL_DEPTH_TEST); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVMap::toggle_map
 *
 *****************************************************************************/
void PVGL::PVMap::toggle_map()
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!visible) {
		visible = true;
		// Set fbos dirty because these informations have been discarded if the map
		// wasn't visible. See ticket # for more information
		set_lines_fbo_dirty();
		set_zombie_fbo_dirty();
	}
	else {
		visible = false;
	}
}

/******************************************************************************
 *
 * PVGL::PVMap::mouse_down
 *
 *****************************************************************************/
bool PVGL::PVMap::mouse_down(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!visible || !picviz_view) {
		return false;
	}
	y = view->get_height() - y;
	if (x > allocation.x && x < allocation.x + allocation.width &&
	    y > allocation.y && y < allocation.y + allocation.height) {
		old_mouse_x = x;
		old_mouse_y = y;

		float min_x = -0.5f;
		float min_y = -0.1f;
		float max_x = picviz_view->get_axes_count() - 0.5f;
		float max_y = 1.1f;

		vec2 factor;
		factor.x = allocation.width / (max_x - min_x);
		factor.y = allocation.height / (max_y - min_y);

		min_x = allocation.x;
		min_y = allocation.y;
		max_x = allocation.x + allocation.width;
		max_y = allocation.y + allocation.height;


		vec2 viewed_min, viewed_max;
		viewed_min.x = (min_x + max_x) / 2.0 - allocation.width / 2.0 / pow(1.2, view->zoom_level_x) - view->translation.x * factor.x;
		viewed_min.y = (min_y + max_y) / 2.0 - allocation.height/ 2.0 / pow(1.2, view->zoom_level_y) - (view->translation.y - 0.1) * factor.y;
		viewed_max.x = (min_x + max_x) / 2.0 + allocation.width / 2.0 / pow(1.2, view->zoom_level_x) - view->translation.x * factor.x;
		viewed_max.y = (min_y + max_y) / 2.0 + allocation.height/ 2.0 / pow(1.2, view->zoom_level_y) - (view->translation.y - 0.1) * factor.y;

		//std::cout << "min: (" << viewed_min.x << ", " << viewed_min.y << "), max: (" << viewed_max.x << ", " << viewed_max.y << "), mouse: (" << x << ", " << y << "), translation: (" << view->translation.x << ", " << view->translation.y << ")" << std::endl;
		if (x > viewed_min.x && x < viewed_max.x &&
		    y > viewed_min.y && y < viewed_max.y) {
			//std::cout << "Click in view!" << std::endl;
			move_view_mode = true;
			dragging = true;
			lines->reset_offset();
		}
		grabbing = true;
		return true;
	} else if (x > allocation.x - 6 && x < allocation.x + allocation.width + 6 &&
	           y > allocation.y + allocation.height && y < allocation.y + allocation.height + 26) {
		old_mouse_x = x;
		old_mouse_y = y;
		dragging = true;
		move_view_mode = false;
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVMap::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVMap::mouse_move(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	y = view->get_height() - y;
	if (dragging) {
		if (move_view_mode) {
			view->translation.x -= (x - old_mouse_x) * float(picviz_view->get_axes_count()) / allocation.width;
			view->translation.y -= (y - old_mouse_y) * 1.2f / allocation.height;
			lines->move_offset(vec2((old_mouse_x - x) * float(view->get_width()) / float(allocation.width) * picviz_view->get_axes_count() / (view->xmax - view->xmin) * pow(1.2, view->zoom_level_x),
			                        (y - old_mouse_y) * float(view->get_height()) / float(allocation.height) * 1.2f / (view->ymax - view->ymin) * pow(1.2, view->zoom_level_y) ));
			old_mouse_x = x;
			old_mouse_y = y;
			//			XXX we might want this to be updated in live, if there are few lines.
//			lines->set_main_fbo_dirty();
//			lines->set_zombie_fbo_dirty();
		} else {
			allocation.x += x - old_mouse_x;
			allocation.y += y - old_mouse_y;
			old_mouse_x = x;
			old_mouse_y = y;
		}
		PVGL::wtk_window_need_redisplay();
		return true;
	} else if (grabbing) {
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVMap::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVMap::mouse_up(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVMap::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	y = view->get_height() - y;
	if (dragging) {
		if (move_view_mode) {
			view->translation.x -= (x - old_mouse_x) * float(picviz_view->get_axes_count()) / allocation.width;
			view->translation.y -= (y - old_mouse_y) * 1.2f / allocation.height;
			old_mouse_x = x;
			old_mouse_y = y;
			lines->set_main_fbo_dirty();
			lines->set_zombie_fbo_dirty();
			lines->reset_offset();

			move_view_mode = false;
		} else {
			allocation.x += x - old_mouse_x;
			allocation.y += y - old_mouse_y;
			PVGL::wtk_window_need_redisplay();
		}
		dragging = false;
		grabbing = false;
		return true;
	} else if (grabbing) {
		grabbing = false;
		return true;
	}
	return false;
}
