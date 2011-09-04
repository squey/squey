#include <QString>

#include <pvsdk/PVMessenger.h>

#include <pvgl/PVUtils.h>

#include <pvgl/PVMain.h>

#include <WtkInit.h>

static PVSDK::PVMessenger *pvsdk_messenger = 0;
static bool _should_stop = false;

bool pvgl_init(PVSDK::PVMessenger *messenger)
{
	int argc = 1;
	char *argv[] = { const_cast<char*>("PVGL"), NULL };
	pvsdk_messenger = messenger;

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
		PVSDK::PVMessage message;
		if (messenger->get_message_for_gl(message)) {
			switch (message.function) {
				case PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT:
				{
					PVLOG_INFO("PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT:\n");
			
								// QString *name = reinterpret_cast<QString *>(message.pointer_1);
								// PVGL::PVMain::create_view(name);
								// //message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
								// //pvgl_com->post_message_to_qt(message);
								// glutTimerFunc(5/*20*/, PVGL::PVMain::timer_func, 0);
								// glutMainLoop();

								// PVGL::wtk_init(argc, argv);
				}
				break;
				case PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT:
				{
					PVLOG_INFO("PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT:\n");
								// if (transient_view) {
								// 	glutDestroyWindow(transient_view->get_window_id());
								// 	delete transient_view;
								// 	transient_view = 0;
								// }
				}
				break;
				case PVSDK_MESSENGER_FUNCTION_CREATE_VIEW:
				{
					PVLOG_INFO("PVSDK_MESSENGER_FUNCTION_CREATE_VIEW:\n");
					
								// all_drawables.push_back(transient_view);
								// glutSetWindow(transient_view->get_window_id());
								// transient_view->init(message.pv_view);
								// message.pointer_1 = new QString(transient_view->get_name());
								// transient_view = 0;
								// message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
								// pvgl_com->post_message_to_qt(message);
								// glutTimerFunc(5/*20*/, PVGL::PVMain::timer_func, 0);
								// glutMainLoop();

								// PVGL::wtk_init(argc, argv);
				}
				break;
				case PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW:
				{
					PVLOG_INFO("PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW:\n");
								// QString *name = reinterpret_cast<QString *>(message.pointer_1);
								// PVGL::PVMain::create_scatter(name, message.pv_view);
								// PVLOG_INFO("PVGL::%s scatter view created\n", __FUNCTION__);
								// message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
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
	PVSDK::PVMessage message;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	if (_should_stop) {
		PVLOG_ERROR("PVGL::PVMain::%s: we are exiting, don't do too much!\n", __FUNCTION__);
		return;
	}

	if (pvsdk_messenger->get_message_for_gl(message)) {
		switch (message.function) {
		case PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT:
			{
			QString *name = reinterpret_cast<QString *>(message.pointer_1);
			PVLOG_ERROR("We shall create our parallel view now!\n");
			// we create our view here
			}
			break;
		case PVSDK_MESSENGER_FUNCTION_SELECTION_CHANGED:
		case PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING:
		case PVSDK_MESSENGER_FUNCTION_CLEAR_SELECTION:
		case PVSDK_MESSENGER_FUNCTION_SCREENSHOT_CHOOSE_FILENAME:
		case PVSDK_MESSENGER_FUNCTION_SCREENSHOT_TAKEN:
		case PVSDK_MESSENGER_FUNCTION_ONE_VIEW_DESTROYED:
		case PVSDK_MESSENGER_FUNCTION_VIEWS_DESTROYED:
		case PVSDK_MESSENGER_FUNCTION_VIEW_CREATED:
		case PVSDK_MESSENGER_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER:
		case PVSDK_MESSENGER_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER:
		case PVSDK_MESSENGER_FUNCTION_SET_COLOR:
		case PVSDK_MESSENGER_FUNCTION_CREATE_VIEW:
		case PVSDK_MESSENGER_FUNCTION_REFRESH_VIEW:
		case PVSDK_MESSENGER_FUNCTION_TAKE_SCREENSHOT:
		case PVSDK_MESSENGER_FUNCTION_DESTROY_VIEWS:
		case PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW:
		case PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT:
		case PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW:
		case PVSDK_MESSENGER_FUNCTION_TOGGLE_DISPLAY_EDGES:
		case PVSDK_MESSENGER_FUNCTION_SET_VIEW_WINDOWTITLE:
		case PVSDK_MESSENGER_FUNCTION_UPDATE_OTHER_SELECTIONS:
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
