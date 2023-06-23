import os
import zipfile

_P4_FILES = [
    "pose_cnn.py",
    "pose_estimation.ipynb"
]


def make_p4_submission(assignment_path, uniquename=None, umid=None):
    _make_submission(assignment_path, _P4_FILES, "P4", uniquename, umid)


def _make_submission(
    assignment_path, file_list, assignment_no, uniquename=None, umid=None
):
    if uniquename is None or umid is None:
        uniquename, umid = _get_user_info()
    zip_path = "{}_{}_{}.zip".format(uniquename, umid, assignment_no)
    zip_path = os.path.join(assignment_path, zip_path)
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            if filename.startswith('rob599/'):
                filename_out = filename.split('/')[-1]
            else:
                filename_out = filename
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename_out)


def _get_user_info():
    uniquename = None
    umid = None
    if uniquename is None:
        uniquename = input("Enter your uniquename (e.g. topipari): ")
    if umid is None:
        umid = input("Enter your umid (e.g. 12345678): ")
    return uniquename, umid
