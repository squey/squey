#include <QString>

#include <pvgl/PVCom.h>

#include <pvgl/PVMain.h>

static PVGL::PVCom *pvgl_com = 0;
static bool _should_stop = false;


void PVGL::PVMain::timer_func(int)
{
	PVGL::PVMessage message;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	if (_should_stop) {
		PVLOG_ERROR("PVGL::PVMain::%s: we are exiting, don't do too much!\n", __FUNCTION__);
		return;
	}

	if (pvgl_com->get_message_for_gl(message)) {
		switch (message.function) {
		case PVGL_COM_FUNCTION_PLEASE_WAIT:
			{
			QString *name = reinterpret_cast<QString *>(message.pointer_1);
			PVLOG_INFO("We shall create our parallel view now!\n");
			// we create our view here
			}
			break;
		case PVGL_COM_FUNCTION_SELECTION_CHANGED:
		case PVGL_COM_FUNCTION_REFRESH_LISTING:
		case PVGL_COM_FUNCTION_CLEAR_SELECTION:
		case PVGL_COM_FUNCTION_SCREENSHOT_CHOOSE_FILENAME:
		case PVGL_COM_FUNCTION_SCREENSHOT_TAKEN:
		case PVGL_COM_FUNCTION_ONE_VIEW_DESTROYED:
		case PVGL_COM_FUNCTION_VIEWS_DESTROYED:
		case PVGL_COM_FUNCTION_VIEW_CREATED:
		case PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER:
		case PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER:
		case PVGL_COM_FUNCTION_SET_COLOR:
		case PVGL_COM_FUNCTION_CREATE_VIEW:
		case PVGL_COM_FUNCTION_REFRESH_VIEW:
		case PVGL_COM_FUNCTION_TAKE_SCREENSHOT:
		case PVGL_COM_FUNCTION_DESTROY_VIEWS:
		case PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW:
		case PVGL_COM_FUNCTION_DESTROY_TRANSIENT:
		case PVGL_COM_FUNCTION_REINIT_PVVIEW:
		case PVGL_COM_FUNCTION_TOGGLE_DISPLAY_EDGES:
		case PVGL_COM_FUNCTION_SET_VIEW_WINDOWTITLE:
		case PVGL_COM_FUNCTION_UPDATE_OTHER_SELECTIONS:
			break;
		default:
			PVLOG_ERROR("PVGL::%s unknown function in a message: %d\n", __FUNCTION__, message.function);
		}
	}

}
