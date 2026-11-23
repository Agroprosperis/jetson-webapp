import threading

import auth
import dashboard_settings
import db
import ownership
import roboflow_settings
import smtp_settings


_init_lock = threading.Lock()


def init_storage():
    with _init_lock:
        connection = db.connect()
        try:
            auth.init_schema(connection)
            ownership.init_schema(connection)
            dashboard_settings.init_schema(connection)
            roboflow_settings.init_schema(connection)
            smtp_settings.init_schema(connection)
            connection.commit()
        finally:
            connection.close()

        roboflow_settings.initialize_roboflow_token_storage()
