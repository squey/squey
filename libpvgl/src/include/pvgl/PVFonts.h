//! \file PVFonts.h
//! $Id: PVFonts.h 2986 2011-05-26 09:51:13Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_FONTS_H
#define LIBPVGL_FONTS_H

#include <map>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <ft2build.h>
#include FT_FREETYPE_H

namespace PVGL {
//! A structure containing all the data allowing to cache a glyph into a texture.
struct LibGLDecl PVGlyph {
	GLfloat uv[4][2];  //!< The uv coordinates of the glyph within the textures.
	int     width;     //!< The width of the glyph in pixels.
	int     height;    //!< The height of the glyph in pixels.
	int     left;      //!< Amount of horizontal space to left at the beginning of the glyph.
	int     top;       //!< Amount of vertical space to left at the beginning of the glyph.
	int     advance_x; //!< Amount of pixels to advance to begin the next glyph rendering.
	int     advance_y; //!< Amount of pixels to advance to begin the next glyph rendering.
};

/**
 * \class PVFont
 *
 * \brief Our font rendering system.
 */
class LibGLDecl PVFont {
	struct GlyphIndex {
		int size;
		int index;
		bool operator!=(const GlyphIndex&gi)const{return size!=gi.size||index!=gi.index;}
		bool operator<(const GlyphIndex&gi)const{return size<gi.size||(size==gi.size&&index<gi.index);}
		GlyphIndex(int s, int i):size(s),index(i){}
	};
	FT_Face                face;        //!< The font face (directly related to the ttf font file, once loaded).
	GLuint                 texture;     //!< The texture containing all the currently known glyphes.
	std::map<GlyphIndex, PVGlyph> glyph_cache; //!< A cache for all the glyphes we already has rendered.
	int                    x_off;       //!< The x position in the texture of the next rendered glyph.
	int                    y_off;       //!< The y position in the texture of the next rendered glyph.
	int                    y_max;       //!< The maximum height of glyphes in the current row in the texture.
public:
	/**
	 * Constructor.
	 */
	PVFont ();

	/**
	 * Destructor.
	 */
	~PVFont ();

	/**
	 * \brief Draw a text at the given position on screen.
	 *
	 * @param x          The x position of the base of the first character (in pixels)
	 * @param y          The y position of the base of the first character (in pixels)
	 * @param text       The text to be displayed.
	 * @param font_size  The size of the font, typically between 10 and 22.
	 */
	void draw_text (float x, float y, const char *text, int font_size);

	/**
	 * @param[in]  text          The text whose size has to be computed
	 * @param[in]  font_height   The font height that will be used to compute the size of the text
	 * @param[out] string_width  The resulting width
	 * @param[out] string_height The resulting height
	 * @param[out] string_ascent The resulting ascent
	 */
	void get_text_size(const std::string &text, int font_height, int &string_width, int &string_height, int &string_ascent);
};

}

#endif
