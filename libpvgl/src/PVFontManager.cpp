//! PVFontManager.cpp
//! $Id: PVFontManager.cpp 2986 2011-05-26 09:51:13Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVUtils.h>
#include <pvgl/PVFonts.h>

const int TEXTURE_FONT_SIZE = 2048;
const int TEXTURE_FONT_PADDING = 3;

/******************************************************************************
 *
 * PVGL::PVFont::PVFont
 *
 ******************************************************************************/
PVGL::PVFont::PVFont()
{
	FT_Error    error;
	FT_Library  library;
	GLubyte    *empty_data;

	PVLOG_INFO("PVGL::PVFont::%s\n", __FUNCTION__);

	error = FT_Init_FreeType(&library);
	if (error) {
		PVLOG_INFO("PVGL::PVFont::%s: Cannot open the freetype library\n", __FUNCTION__);
	}
	error = FT_New_Face(library, (pvgl_get_share_path() + "FreeSerif.ttf").c_str(), 0, &face);
	if (error) {
		PVLOG_INFO("PVGL::PVFont::%s: Cannot load the %s font file.\n", "FreeSerif.ttf");
	}
	error = FT_Select_Charmap(face, FT_ENCODING_UNICODE);
	if (error) {
		PVLOG_INFO("PVGL::PVFont::%s: Cannot find the Unicode encoding in the current font.\n", __FUNCTION__);
	}
	error = FT_Set_Char_Size(face, 22 * 64, 22 * 64, 100, 100);
	if (error) {
		PVLOG_INFO("PVGL::PVFont::%s: Cannot select a char size of %d in the current font.\n", __FUNCTION__, 22);
	}

	glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glGenTextures(1, &texture);
	glBindTexture (GL_TEXTURE_2D, texture);
	empty_data = new GLubyte[TEXTURE_FONT_SIZE * TEXTURE_FONT_SIZE];
	memset(empty_data, 0, TEXTURE_FONT_SIZE * TEXTURE_FONT_SIZE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA,
	             TEXTURE_FONT_SIZE, TEXTURE_FONT_SIZE, 0,
	             GL_ALPHA, GL_UNSIGNED_BYTE, empty_data);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	x_off = y_off = y_max = 0;
	delete[] empty_data;
}

/******************************************************************************
 *
 * PVGL::PVFont::draw_text
 *
 ******************************************************************************/
void PVGL::PVFont::draw_text(float x, float y, const char *text, int font_size)
{
	PVLOG_HEAVYDEBUG("PVGL::PVFont::%s\n", __FUNCTION__);

	FT_Error    error;
	FT_GlyphSlot slot = face->glyph;

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	error = FT_Set_Char_Size(face, font_size * 64, font_size * 64, 100, 100);
	while (*text) {
		PVGlyph glyph;
		int       c;
		/* set transformation */
		// Load glyph image into the slot (erase previous one).
		c = pvgl_get_next_utf8(text);
		// If found in the cache, use it!
		if (glyph_cache.find(GlyphIndex(font_size, c)) != glyph_cache.end ()) {
			glyph = glyph_cache[GlyphIndex(font_size, c)];
		} else { // Else create it!
			error = FT_Load_Char(face, c, FT_LOAD_RENDER);
			glyph.width  = slot->bitmap.width;
			glyph.height = slot->bitmap.rows;
			//
			y_max = std::max(y_max, glyph.height);
			if (x_off + glyph.width > TEXTURE_FONT_SIZE) {
				y_off += y_max + TEXTURE_FONT_PADDING;
				if (y_off > TEXTURE_FONT_SIZE) {
					PVLOG_ERROR("Font cache is bigger than the texture. Increase the texture size or handle multiple textures.\n");
					PVLOG_ERROR("%d glyphes rendered.\n", glyph_cache.size());
					for (std::map<GlyphIndex,PVGlyph>::iterator it = glyph_cache.begin(); it != glyph_cache.end(); ++it) {
						PVLOG_ERROR("Glyph: %d\n", it->first.index);
					}
				}
				x_off = 0;
			}
			// Render into the texture
			glTexSubImage2D(GL_TEXTURE_2D, 0,
			                x_off, y_off,
			                slot->bitmap.width, slot->bitmap.rows,
			                GL_ALPHA, GL_UNSIGNED_BYTE,
			                slot->bitmap.buffer);
			glyph.uv[0][0] =  x_off                 / float(TEXTURE_FONT_SIZE);
			glyph.uv[0][1] =  y_off                 / float(TEXTURE_FONT_SIZE);
			glyph.uv[1][0] = (x_off + glyph.width ) / float(TEXTURE_FONT_SIZE);
			glyph.uv[1][1] =  y_off                 / float(TEXTURE_FONT_SIZE);
			glyph.uv[2][0] = (x_off + glyph.width ) / float(TEXTURE_FONT_SIZE);
			glyph.uv[2][1] = (y_off + glyph.height) / float(TEXTURE_FONT_SIZE);
			glyph.uv[3][0] =  x_off                 / float(TEXTURE_FONT_SIZE);
			glyph.uv[3][1] = (y_off + glyph.height) / float(TEXTURE_FONT_SIZE);
			glyph.top  = slot->bitmap_top;
			glyph.left = slot->bitmap_left;
			glyph.advance_x = slot->advance.x / 64;
			glyph.advance_y = slot->advance.y / 64;
			x_off += glyph.width + TEXTURE_FONT_PADDING;
			glyph_cache[GlyphIndex(font_size, c)] = glyph;
		}
		// The real rendering.
		glBegin(GL_QUADS);
		glTexCoord2fv(&glyph.uv[0][0]); glVertex2f(x,               y - glyph.top);
		glTexCoord2fv(&glyph.uv[1][0]); glVertex2f(x + glyph.width, y - glyph.top);
		glTexCoord2fv(&glyph.uv[2][0]); glVertex2f(x + glyph.width, y - (glyph.top - glyph.height));
		glTexCoord2fv(&glyph.uv[3][0]); glVertex2f(x,               y - (glyph.top - glyph.height));
		glEnd();
		x += glyph.advance_x;
	}
	/* The following is useful for debugging purpose:  */
#if 0
	glBegin(GL_QUADS);
		glTexCoord2f(0,0); glVertex2f(0,0);
		glTexCoord2f(0,1); glVertex2f(0,TEXTURE_FONT_SIZE);
		glTexCoord2f(1,1); glVertex2f(TEXTURE_FONT_SIZE,TEXTURE_FONT_SIZE);
		glTexCoord2f(1,0); glVertex2f(TEXTURE_FONT_SIZE,0);
	glEnd();
#endif
	// Restore a normal state.
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
}

/******************************************************************************
 *
 * PVGL::PVFont::get_text_size
 *
 ******************************************************************************/
void PVGL::PVFont::get_text_size(const std::string &text, int font_height, int &string_width, int &string_height, int &string_ascent)
{
	FT_Error    error;
	int         index;
	FT_Vector   pen;
	int         ascent, descent;
	const char *c_text = text.c_str();

	PVLOG_HEAVYDEBUG("PVGL::PVFont::%s\n", __FUNCTION__);

	FT_Set_Char_Size(face, font_height << 6, font_height << 6, 100, 100);

	ascent = descent = 0;
	pen.x = pen.y = 0;

	while (*c_text) {
		int c;

		FT_Set_Transform(face, 0, &pen);
		c	= pvgl_get_next_utf8(c_text);
		index = FT_Get_Char_Index(face, c);
		error = FT_Load_Glyph(face, index, 0);
		if (face->glyph->format != FT_GLYPH_FORMAT_BITMAP) {
			error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
		}
		ascent = std::max(ascent, face->glyph->bitmap_top);
		descent= std::max(descent,face->glyph->bitmap.rows - face->glyph->bitmap_top);
		pen.x += face->glyph->advance.x;
		pen.y += face->glyph->advance.y;

	}
	string_width = pen.x >> 6;
	string_height = ascent + descent;
	string_ascent = ascent;
}
