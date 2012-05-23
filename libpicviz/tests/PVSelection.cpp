#include <stdlib.h>

#include <picviz/PVSelection.h>

#include <iostream>


int main(void)
{

// #include "test-env.h"

	Picviz::PVSelection *selection;
	Picviz::PVSelection *selection2 = new Picviz::PVSelection();
	Picviz::PVSelection *selection3 = new Picviz::PVSelection();
	Picviz::PVSelection a;
	Picviz::PVSelection b;
	
	

	int i, good;
	int line_index;

	PVRow last_index;
	PVRow count;

	/**********************************************************************
	*
	* We test creation and deletion of a picviz_selection_t
	*
	**********************************************************************/

	selection = new Picviz::PVSelection();
	delete(selection);



	/**********************************************************************
	*
	* We test all Generic functions
	*
	**********************************************************************/

	std::cout << "we test get_number_of_selected_lines_in_range() and select_all()\n";
	last_index = PICVIZ_LINES_MAX;

	selection = new Picviz::PVSelection();
	selection->select_all();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << PICVIZ_LINES_MAX << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX) {
		return 1;
	}

	std::cout << "we test select_even()\n";
	selection->select_even();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << PICVIZ_LINES_MAX/2 << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX/2) {
		return 1;
	}
	
	std::cout << "we test select_odd()\n";
	selection->select_odd();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << PICVIZ_LINES_MAX/2 << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX/2) {
		return 1;
	}

	std::cout << "we test get_number_of_selected_lines_in_range(a,b)\n";
	count = selection->get_number_of_selected_lines_in_range(0, 100);
	std::cout << "is 50 = to " << count << " ?\n\n";
	if (count != 50) {
		return 1;
	}

	// Test of C++0x features
	a.select_even();
	b.select_odd();

	Picviz::PVSelection c = a | b;
	PVLOG_INFO("a: %p , b = %p , c = %p\n", &a, &b, &c);
	std::cout << "PVSelection should be empty: PVSelection::is_empty() = " << c.is_empty() << std::endl;

	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_A2A_****
	*
	***********************************************************************
	**********************************************************************/


	/**********************************************************************
	*
	* We test picviz_selection_A2A_inverse()
	*
	**********************************************************************/

	std::cout << "\nWe test the operator~\n";
	b.select_all();
	a = ~b;

	for (i=0; i<PICVIZ_LINES_MAX; i++) {
		good = (a.get_line(i));
		if (good) {
			std::cout << " i = " << i << ", and state of line n is : " << good << "\n";
			std::cout << "selection test : [" << __LINE__ << "] : operator~ : failed\n";
			return 1;
		}
	}





	/**********************************************************************
	*
	* We delete remaining objects
	*
	**********************************************************************/
	//delete(selection);
	delete(selection2);
	delete(selection3);

#if 0
	Picviz::PVSelection *a = new Picviz::PVSelection();
	Picviz::PVSelection b;
	a->select_all();
	b.select_odd();
	*a = ~b;
	delete a;
#endif

	return 0;


	std::cout << "#### MAPPED ####\n";

}
