//! \file PVCom.h
//! $Id: PVCom.h 3120 2011-06-14 05:16:22Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011


#ifndef PVGL_COM_H
#define PVGL_COM_H

#include <queue>

#include <QtCore>

#include <picviz/PVView.h>

namespace PVGL {
class PVCom;
}
#include <pvgl/general.h>

/**
 * \enum PVGLComFunction. All the message functions we're able to handle.
 */
enum PVGLComFunction
{
	// First the fonctions asked by the PVGL view to the Qt interface.
	PVGL_COM_FUNCTION_SELECTION_CHANGED = 601,           //!< The selection has changed, the Qt interface should update the listing and other stuff.
	PVGL_COM_FUNCTION_REFRESH_LISTING = 602,             //!< The Qt interface should refresh the listing view.
	PVGL_COM_FUNCTION_CLEAR_SELECTION = 603,             //!< The selection has been cleared.
	PVGL_COM_FUNCTION_SCREENSHOT_CHOOSE_FILENAME = 604,  //!< The Qt interface should ask for a screenshot filename.
	PVGL_COM_FUNCTION_SCREENSHOT_TAKEN = 605,            //!< A screenshot has been saved, the name can be freed by the Qt interface.
	PVGL_COM_FUNCTION_ONE_VIEW_DESTROYED = 606,          //!< One PVGL::PVView has been destroyed, inform the Main Interface about this.
	PVGL_COM_FUNCTION_VIEWS_DESTROYED = 607,             //!< All the PVGL views of a Picviz::PVView have been destroyed, inform the Main Interface about this so it can destroy the Picviz::PVView too.
	PVGL_COM_FUNCTION_VIEW_CREATED = 608,                //!< The PVGL::PVView has been created, the Main Interface can now free the name of the view.
	PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER = 609, //!< The user has commited a changed in the current layer, tell everybody about this.
	PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER = 610, //!< The user has commited a changed in a new layer, tell everybody about this.
	PVGL_COM_FUNCTION_SET_COLOR = 611,                     //!< Show a color-choosing dialog box to color the current selection.

	// Then the fonctions asked by the Qt interface to the PVGL view.
	PVGL_COM_FUNCTION_PLEASE_WAIT = 701,          //!< The PVGL should display a waiting message during the loading of a file.
	PVGL_COM_FUNCTION_CREATE_VIEW = 702,          //!< The PVGL should create a new view.
	PVGL_COM_FUNCTION_REFRESH_VIEW = 703,         //!< The PVGL should refresh the view, the lines may have different properties.
	PVGL_COM_FUNCTION_TAKE_SCREENSHOT = 704,      //!< The PVGL should capture the current view and save it in the provided filename.
	PVGL_COM_FUNCTION_DESTROY_VIEWS = 705,        //!< The Main Interface (Qt) has destroyed a Picviz::PVView, let's destroy all the associated views.
	PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW = 706,  //!< The PVGL should create a new, scattered view.
	PVGL_COM_FUNCTION_DESTROY_TRANSIENT = 707,    //!< The PVGL destroy the current transient view.
	PVGL_COM_FUNCTION_REINIT_PVVIEW = 708,        //!< The Picviz view has changed, everything should be recreated.
	PVGL_COM_FUNCTION_TOGGLE_DISPLAY_EDGES = 709, //!< We click on the menu Axes/Display/Hide Edges
	PVGL_COM_FUNCTION_SET_VIEW_WINDOWTITLE = 710, //!< We update the title of the view window

	// Then functions asked by PVGL for PVGL
	PVGL_COM_FUNCTION_UPDATE_OTHER_SELECTIONS = 801 //!< The PVGL should update the selection in other views
};


/**
 * \enum PVGLComRefreshStates. Which parts should be refreshed.
 */
enum PVGLComRefreshStates
{
  PVGL_COM_REFRESH_POSITIONS =  1,
  PVGL_COM_REFRESH_Z         =  2,
	PVGL_COM_REFRESH_COLOR     =  4,
	PVGL_COM_REFRESH_ZOMBIES   =  8,
	PVGL_COM_REFRESH_SELECTION = 16,
	PVGL_COM_REFRESH_AXES      = 32
};


/**
 *  \page messages Messages System
 *
 *  The Currently supported messages are:
 *  <ul>
 *    <li>From the PVGL to the Main Interface (Qt)
 *      <ul>
 *        <li>#PVGL_COM_FUNCTION_SELECTION_CHANGED<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_REFRESH_LISTING<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_CLEAR_SELECTION<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_SCREENSHOT_CHOOSE_FILENAME<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>int_1: the window_id</li>
 *              <li>int_2: the modifiers (alt, shift, controls, etc.) (unused).</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_SCREENSHOT_TAKEN<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>pointer_1: filename</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_ONE_VIEW_DESTROYED<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>pointer_1: the name of the view (QString *)</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_VIEWS_DESTROYED<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_VIEW_CREATED<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>pointer_1: the name of the view (QString *)</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *      </ul>
 *    </li>
 *    <li>From the Main Interface to the PVGL
 *      <ul>
 *        <li>#PVGL_COM_FUNCTION_PLEASE_WAIT<br />
 *            Parameters:
 *            <ul>
 *              <li>int_1: waiting state (from 1 to 7)</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_CREATE_VIEW<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>pointer_1: filename of the view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_REFRESH_VIEW<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>int_1: bitfield telling which part should be refreshed, see #PVGLComRefreshStates.</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_TAKE_SCREENSHOT<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>int_1: window_id</li>
 *              <li>pointer_1: filename</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_DESTROY_VIEWS<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>pointer_1: filename of the view</li>
 *            </ul>
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_DESTROY_TRANSIENT<br />
 *            Parameters: None.
 *        </li>
 *        <li>#PVGL_COM_FUNCTION_REINIT_PVVIEW<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *            </ul>
 *        </li>
 *      </ul>
 *    </li>
 *    <li>From the PVGL to itself.
 *      <ul>
 *        <li>#PVGL_COM_FUNCTION_UPDATE_OTHER_SELECTIONS<br />
 *            Parameters:
 *            <ul>
 *              <li>pv_view</li>
 *              <li>int_1: the window_id of the view issuing the message (not to be updated again)</li>
 *            </ul>
 *        </li>
 *      </ul>
 *    </li>
 *  </ul>
 */

namespace PVGL {

/**
 * \struct PVMessage
 *
 * The structure describing a message. Fields don't need to always be filled.
 */
struct LibExport PVMessage
{
	PVGLComFunction  function;  //!< The type of the message (always a function) see #PVGLComFunction.
	Picviz::PVView_p pv_view;   //!< A pointer to the Picviz::PVView the message is about.

	int    int_1;               //!< First integer parameter for the message, if needed.
	int    int_2;               //!< Second integer parameter for the message, if needed.
	float  float_1;             //!< First float parameter for the message, if needed.
	float  float_2;             //!< Second float parameter for the message, if needed.
	void  *pointer_1;           //!< First pointer parameter for the message, if needed.
	// ...
};

/**
 * \class PVCom
 */
class LibExport PVCom
{
	QMutex mutex;                     //!< A Mutex protecting the access to all our message queues.
	std::queue<PVMessage> gl_to_qt; //!< The queue to store all the messages from the PVGL, waiting for the Main Interface to handle them.
	std::queue<PVMessage> qt_to_gl; //!< The queue to store all the messages from the Main Interface, waiting for the PVGL to handle them.
public:
	/** Check if a new message has been posted by the Main Interface for the PVGL to handle.
	 *
	 * @param[out] message If a message has been received, it will be placed in this place-hodler.
	 *
	 * @return true if a message has been received, false otherwise.
	 */
	bool get_message_for_gl(struct PVMessage &message);

	/** Check if a new message has been posted by the PVGL for the Main Interface to handle.
	 *
	 * @param[out] message If a message has been received, it will be placed in this place-hodler.
	 *
	 * @return true if a message has been received, false otherwise.
	 */
	bool get_message_for_qt(struct PVMessage &message);

	/**
	 *
	 * @param[in] message The message to be posted.
	 */
	void post_message_to_gl(const struct PVMessage &message);

	/**
	 *
	 * @param[in] message The message to be posted.
	 */
	void post_message_to_qt(const struct PVMessage &message);
};

/**
 * \class PVThread
 *
 * The main thread of the PVGL.
 */
class LibExport PVThread : public QThread
{
	PVGL::PVCom *pvgl_com; //!<
public:
	/**
	 *  Constructor.
	 */
	PVThread () {pvgl_com = new PVGL::PVCom;}

	/**
	 * @return
	 */
	PVGL::PVCom *get_com(){return pvgl_com;}

	/**
	 *
	 */
	void run();

};
}
#endif
