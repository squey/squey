//! \file PVUtils.h
//! $Id: PVUtils.h 2872 2011-05-19 03:30:31Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_UTILS_H
#define LIBPVGL_UTILS_H

#include <QtCore>

#include <vector>
#include <string>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/gl.h>

#include <pvkernel/core/general.h>

//!
#define BUFFER_OFFSET(bytes) ((GLubyte*) NULL + (bytes))

#ifndef NDEBUG
//!
#define PRINT_OPENGL_ERROR() print_opengl_error(__FILE__, __LINE__)
#else
#define PRINT_OPENGL_ERROR() do{}while(0)
#endif

/**
 * @param file
 * @param line
 *
 * @return
 */
LibGLDecl bool print_opengl_error(const char *file, int line);

/**
 *
 */
LibGLDecl void check_framebuffer_status(void);

/**
 *
 */
LibGLDecl void get_gl_version(int *major, int *minor);

/**
 *
 */
LibGLDecl void print_shader_info_log(GLuint shader);

/**
 *
 */
LibGLDecl void print_program_info_log(GLuint program);

/**
 *
 */
LibGLDecl GLint get_uni_loc(GLuint program, const GLchar *name);

/**
 *
 */
LibGLDecl GLuint read_shader(const std::string              &vertex_filename,
                   const std::string              &geometry_filename,
                   const std::string              &fragment_filename,
                   const std::string              &vertex_prefix,
                   const std::string              &geometry_prefix,
                   const std::string              &fragment_prefix,
                   const std::vector<std::string> &attributes);

/**
 *
 */
LibGLDecl void fixing_glew_bugs(void);

/**
 *
 */
LibGLDecl std::string pvgl_get_share_path();

/**
 * @return true if the pvgl share path is found. False otherwise.
 */
LibGLDecl bool pvgl_share_path_exists();

/**
 * @param text
 */
LibGLDecl int pvgl_get_next_utf8(const char *&text);

#endif
