import joblib


def get_object(object_path):
    """Reads a pkl object into memory


    Args:
        object_path (str): object path
    """

    return joblib.load(object_path)


def save_object(object, object_path):
    """Saves a pkl object into disc


    Args:
        object (Python Object): Pickable python object
        object_path (str): object path
    """

    return joblib.dump(object, object_path)
