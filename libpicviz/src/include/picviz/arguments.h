/**
 * \file arguments.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_ARGUMENTS_H_
#define _PICVIZ_ARGUMENTS_H_

#include <picviz/general.h>

#ifdef __cplusplus
 extern "C" {
#endif

enum _picviz_argument_type_t {
	PICVIZ_ARGUMENT_TYPE_NONE,
	PICVIZ_ARGUMENT_STRING,
	PICVIZ_ARGUMENT_INT,
	PICVIZ_ARGUMENT_FLOAT,
	PICVIZ_ARGUMENT_DOUBLE,
	PICVIZ_ARGUMENT_LAYER,
	PICVIZ_ARGUMENT_PLOTTED,
};
typedef enum _picviz_argument_type_t picviz_argument_type_t;

enum _picviz_argument_widget_t {
	PICVIZ_ARGUMENT_WIDGET_NONE,
	PICVIZ_ARGUMENT_WIDGET_SINGLESLIDER,
	PICVIZ_ARGUMENT_WIDGET_DUALSLIDER,
	PICVIZ_ARGUMENT_WIDGET_TRIPLESLIDER,
	PICVIZ_ARGUMENT_WIDGET_COLORPICKUP,
	PICVIZ_ARGUMENT_WIDGET_TEXTBOX,
	PICVIZ_ARGUMENT_WIDGET_CHECKBOX,
	PICVIZ_ARGUMENT_WIDGET_AXIS_CHOOSER, /* I want to choose one axis */
	PICVIZ_ARGUMENT_WIDGET_SEVERAL_AXES_CHOOSER, /* I want to choose one or several axes */
};
typedef enum _picviz_argument_widget_t picviz_argument_widget_t;

enum _picviz_argument_slider_position_t {
	PICVIZ_ARGUMENT_SLIDER_POSITION_NONE,
	PICVIZ_ARGUMENT_SLIDER_LEFT,
	PICVIZ_ARGUMENT_SLIDER_CENTER,
	PICVIZ_ARGUMENT_SLIDER_RIGHT,
};
typedef enum _picviz_argument_slider_position_t picviz_argument_slider_position_t;

#define PICVIZ_ARGUMENTS_MAX 1024

#define PICVIZ_ARGUMENT_LIMIT_MAX_DEFAULT 0
#define PICVIZ_ARGUMENT_LIMIT_MIN_DEFAULT 0
#define PICVIZ_ARGUMENT_INCREASE_FACTOR_DEFAULT 0.1
#define PICVIZ_DEFAULT_PARAMETERS 0, 0, 0.1
#define PICVIZ_ARGUMENTS_END {PICVIZ_ARGUMENT_TYPE_NONE, PICVIZ_ARGUMENT_WIDGET_NONE, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0}
#define PICVIZ_ARGUMENTS_STRING_GROUP(name, group, value) {PICVIZ_ARGUMENT_STRING, PICVIZ_ARGUMENT_WIDGET_NONE, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, #name, #group, NULL, value, 0, 0, 0, PICVIZ_DEFAULT_PARAMETERS}
#define PICVIZ_ARGUMENTS_INT(name, value) {PICVIZ_ARGUMENT_INT, PICVIZ_ARGUMENT_WIDGET_NONE, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, #name, NULL, NULL, value, 0, 0, PICVIZ_DEFAULT_PARAMETERS}
#define PICVIZ_ARGUMENTS_TEXTBOX(name, group, value) {PICVIZ_ARGUMENT_STRING, PICVIZ_ARGUMENT_WIDGET_TEXTBOX, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, #name, #group, NULL, value, 0, 0, 0, PICVIZ_DEFAULT_PARAMETERS}
#define PICVIZ_ARGUMENTS_TEXTBOX_AS_FLOAT(name, group, value) {PICVIZ_ARGUMENT_STRING, PICVIZ_ARGUMENT_WIDGET_TEXTBOX, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, #name, #group, NULL, NULL, 0, value, 0, PICVIZ_DEFAULT_PARAMETERS}
#define PICVIZ_ARGUMENTS_SLIDERLEFT(name, group, value) {PICVIZ_ARGUMENT_FLOAT, PICVIZ_ARGUMENT_WIDGET_DUALSLIDER, PICVIZ_ARGUMENT_SLIDER_LEFT, #name, group, NULL, 0, value, 0, 0, 0.0, 1.0, 0.1}
#define PICVIZ_ARGUMENTS_SLIDERRIGHT(name, group, value) {PICVIZ_ARGUMENT_FLOAT, PICVIZ_ARGUMENT_WIDGET_DUALSLIDER, PICVIZ_ARGUMENT_SLIDER_RIGHT, #name, group, NULL, 0, value, 0, 0, 0.0, 1.0, 0.1}
#define PICVIZ_ARGUMENTS_AXIS(name, value) {PICVIZ_ARGUMENT_STRING, PICVIZ_ARGUMENT_WIDGET_AXIS_CHOOSER, PICVIZ_ARGUMENT_SLIDER_POSITION_NONE, #name, NULL, NULL, NULL, value, 0, 0, PICVIZ_DEFAULT_PARAMETERS}

struct _picviz_argument_item_t {
  picviz_argument_type_t type;
  picviz_argument_widget_t widget;
  picviz_argument_slider_position_t slider_position;
  char *name;
  char *group;

  void *elem;
  char *strval;
  int ival;
  float fval;
  double dval;

  float limit_min;
  float limit_max;
  float increase_factor;
};
typedef struct _picviz_argument_item_t picviz_argument_item_t;

struct _picviz_arguments_t {
  picviz_argument_item_t *args;
  int nelts;
  int current_pos;
};
typedef struct _picviz_arguments_t picviz_arguments_t;

LibPicvizDecl picviz_arguments_t *picviz_arguments_new(void);
LibPicvizDecl void picviz_arguments_destroy(picviz_arguments_t *arguments);
LibPicvizDecl int picviz_arguments_item_append(picviz_arguments_t *arguments, picviz_argument_item_t item);
LibPicvizDecl int picviz_arguments_item_list_append(picviz_arguments_t *arguments, picviz_argument_item_t items[]);
LibPicvizDecl void picviz_arguments_item_debug(picviz_argument_item_t item);
LibPicvizDecl void picviz_arguments_debug(picviz_arguments_t *arguments);
LibPicvizDecl picviz_argument_item_t picviz_arguments_item_new(void);
LibPicvizDecl void picviz_arguments_item_destroy(picviz_argument_item_t item);
LibPicvizDecl picviz_argument_item_t picviz_arguments_get_item_from_name(picviz_arguments_t *arguments, const char *name);
LibPicvizDecl void picviz_arguments_set_item_from_name(picviz_arguments_t *arguments, char *name, picviz_argument_item_t item);
LibPicvizDecl picviz_argument_item_t picviz_arguments_get_item_from_group_and_dualslider_position(picviz_arguments_t *arguments, char *group, picviz_argument_slider_position_t position);
LibPicvizDecl char *picviz_arguments_item_get_string(picviz_argument_item_t item);
LibPicvizDecl picviz_argument_item_t picviz_arguments_item_set_string(picviz_argument_item_t item, char *value);
LibPicvizDecl float picviz_arguments_item_get_float(picviz_argument_item_t item);
LibPicvizDecl void picviz_arguments_item_set_float(picviz_argument_item_t item, float value);
LibPicvizDecl float picviz_arguments_read_item_value_as_float(picviz_argument_item_t item);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_ARGUMENTS_H_ */
