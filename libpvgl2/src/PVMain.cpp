#include <QString>

#include <pvgl/PVCom.h>
#include <pvgl/PVUtils.h>

#include <pvgl/PVMain.h>

#include <WtkInit.h>

static PVGL::PVCom *pvgl_com = 0;
static bool _should_stop = false;

bool pvgl_init(PVGL::PVCom *com)
{
	int argc = 1;
	char *argv[] = { const_cast<char*>("PVGL"), NULL };
	pvgl_com = com;

	if (pvgl_share_path_exists() == false) {
		PVLOG_FATAL("Cannot open PVGL share directory %s!\n", pvgl_get_share_path().c_str());
		return false;
	} else {
		PVLOG_INFO("Using PVGL share directory %s\n", pvgl_get_share_path().c_str());
	}

	PVGL::WTK::init(argc, argv);

	// Wait for the first message
	PVLOG_DEBUG("PVGL::%s Everything created, waiting for message.\n", __FUNCTION__);
	for(;;) {
		PVGL::PVMessage message;
		if (com->get_message_for_gl(message)) {
			switch (message.function) {
				case PVGL_COM_FUNCTION_PLEASE_WAIT:
				{
					PVLOG_INFO("PVGL_COM_FUNCTION_PLEASE_WAIT:\n");
			
								// QString *name = reinterpret_cast<QString *>(message.pointer_1);
								// PVGL::PVMain::create_view(name);
								// //message.function = PVGL_COM_FUNCTION_VIEW_CREATED;
								// //pvgl_com->post_message_to_qt(message);
								// glutTimerFunc(5/*20*/, PVGL::PVMain::timer_func, 0);
								// glutMainLoop();

								// PVGL::wtk_init(argc, argv);
				}
				break;
				case PVGL_COM_FUNCTION_DESTROY_TRANSIENT:
				{
					PVLOG_INFO("PVGL_COM_FUNCTION_DESTROY_TRANSIENT:\n");
								// if (transient_view) {
								// 	glutDestroyWindow(transient_view->get_window_id());
								// 	delete transient_view;
								// 	transient_view = 0;
								// }
				}
				break;
				case PVGL_COM_FUNCTION_CREATE_VIEW:
				{
					PVLOG_INFO("PVGL_COM_FUNCTION_CREATE_VIEW:\n");
					
								// all_drawables.push_back(transient_view);
								// glutSetWindow(transient_view->get_window_id());
								// transient_view->init(message.pv_view);
								// message.pointer_1 = new QString(transient_view->get_name());
								// transient_view = 0;
								// message.function = PVGL_COM_FUNCTION_VIEW_CREATED;
								// pvgl_com->post_message_to_qt(message);
								// glutTimerFunc(5/*20*/, PVGL::PVMain::timer_func, 0);
								// glutMainLoop();

								// PVGL::wtk_init(argc, argv);
				}
				break;
				case PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW:
				{
					PVLOG_INFO("PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW:\n");
								// QString *name = reinterpret_cast<QString *>(message.pointer_1);
								// PVGL::PVMain::create_scatter(name, message.pv_view);
								// PVLOG_INFO("PVGL::%s scatter view created\n", __FUNCTION__);
								// message.function = PVGL_COM_FUNCTION_VIEW_CREATED;
								// message.pointer_1 = new QString(*name);
								// pvgl_com->post_message_to_qt(message);
								// glutTimerFunc(5/*20*/, PVGL::PVMain::timer_func, 0);
								// glutMainLoop();

								// PVGL::wtk_init(argc, argv);
				}
				break;
				default:
						PVLOG_ERROR("PVGL::%s unknown function in a message: %d\n", __FUNCTION__, message.function);
			}
		} else {

// Avoids CPU eating
#ifdef WIN32
			Sleep(1000);
#else
			sleep(1);
#endif
		}
	}

	return true;
}

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
			PVLOG_ERROR("We shall create our parallel view now!\n");
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

void PVGL::PVMain::stop()
{
	PVLOG_INFO("PVGL::%s: stopping\n", __FUNCTION__);
	_should_stop = true;
}
