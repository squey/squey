/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVEventline.h>


Inendi::PVEventline::PVEventline(PVRow row_count)
{
	this->row_count = row_count;

	this->first_index = 0;
	this->current_index = row_count - 1;
	this->last_index =  row_count - 1;
}


/******************************************************************************
 *
 * inendi_eventline_get_current_index
 *
 *****************************************************************************/
int Inendi::PVEventline::get_current_index() const
{
	return this->current_index;
}



/******************************************************************************
 *
 * Inendi::PVEventline::get_first_index
 *
 *****************************************************************************/
int Inendi::PVEventline::get_first_index() const
{
	return this->first_index;
}



/******************************************************************************
 *
* Inendi::PVEventline::get_kth_slider_position
 *
 *****************************************************************************/
float Inendi::PVEventline::get_kth_slider_position(int k) const
{
	/* VARIABLES */
	float slider_position = 0;

	/* CODE */
	/* We need to check that we are not in a weird situation where there would be only one row... */
	if (this->row_count > 1) {
		/* There is more than one row : we can compute the float */
		switch (k) {
			case 0:
				slider_position = (float)(this->first_index) / (this->row_count -1);
				break;

			case 1:
				slider_position = (float)(this->current_index) / (this->row_count -1);
				break;

			case 2:
				slider_position = (float)(this->last_index) / (this->row_count -1);
				break;
		}

		return slider_position;
	} else {
		/* There is only one row ! */
		return (float)0.0;
	}
}



/******************************************************************************
 *
 * Inendi::PVEventline::get_last_index
 *
 *****************************************************************************/
int Inendi::PVEventline::get_last_index() const
{
	return this->last_index;
}



/******************************************************************************
 *
 * Inendi::PVEventline::selection_A2A_filter
 *
 *****************************************************************************/
void Inendi::PVEventline::selection_A2A_filter(Inendi::PVSelection &selection)
{
		int i;
		/* We unselect all lines before first_index */
		for (i=0; i<this->first_index; i++) {
			selection.set_line(i, 0);
		}
		/* We unselect all lines after current_index */
		for (i=this->current_index + 1; i<this->row_count; i++) {
			selection.set_line(i, 0);
		}
}



/******************************************************************************
 *
 * Inendi::PVEventline::selection_A2B_filter
 *
 *****************************************************************************/
void Inendi::PVEventline::selection_A2B_filter(Inendi::PVSelection &a, Inendi::PVSelection &b)
{
	b = a;
	selection_A2A_filter(b);
}



/******************************************************************************
 *
 * Inendi::PVEventline::set_current_index
 *
 *****************************************************************************/
void Inendi::PVEventline::set_current_index(int index)
{
	this->current_index = index;
}



/******************************************************************************
 *
 * Inendi::PVEventline::set_first_index
 *
 *****************************************************************************/
void Inendi::PVEventline::set_first_index(int index)
{
	this->first_index = index;
}



/******************************************************************************
 *
 * Inendi::PVEventline::set_kth_index_from_float
 *
 *****************************************************************************/
float Inendi::PVEventline::set_kth_index_and_adjust_slider_position(int k, float x)
{
	/* VARIABLES */
	int real_index;
	float real_position;
	
	/* CODE */
	/* We compute the int position closest the x*row_count */
	real_index = (int)(x*(this->row_count - 1) + 0.5);
	
	/* depending on k, we change the index */
	switch (k) {
		case 0:
			if (real_index < 0) {
				real_index = 0;
			}
			if (real_index > this->current_index) {
				real_index = this->current_index;
			}
			this->first_index = real_index;
			break;

		case 1:
			if (real_index < this->first_index) {
				real_index = this->first_index;
			}
			if (real_index > this->last_index) {
				real_index = this->last_index;
			}
			this->current_index = real_index;
			break;

		case 2:
			if (real_index < this->current_index) {
				real_index = this->current_index;
			}
			if (real_index >= this->row_count) {
				real_index = this->row_count -1;
			}
			this->last_index = real_index;
			break;
	}

	/* Now we adjust the float position of the concerned slider */
	if (this->row_count > 1) {
		real_position = (float)(real_index) / (this->row_count - 1);
		return real_position;
	} else {
		return (float)0.0;
	}
}



/******************************************************************************
 *
 * Inendi::PVEventline::set_last_index
 *
 *****************************************************************************/
void Inendi::PVEventline::set_last_index(int index)
{
	this->last_index = index;
}


int Inendi::PVEventline::get_row_count() const
{
	return row_count;
}