import os
import shutil


def makeDir(path):
    """
    Make the dictionary for the different files.
    
    Parameters
    -------------
    path: str
        Relative path of the files 
    """

    dirNames = ["detections", "images", "statistics"]

    if not os.path.exists(path):
        os.mkdir(path)
    for name in dirNames:
        dst = os.path.join(path, name)
        if not os.path.exists(dst):
            os.mkdir(dst)
        if (name == "detections"):
            src = "/Users/cr22/jjj/1"
            move_files(src, dst)
        elif (name == "images"):
            src = "/Users/cr22/jjj/1"
            move_files(src, dst)
        elif (name == "statistics"):
            src = "/Users/cr22/jjj/1"
            move_files(src, dst)


def move_files(src, dstp):
    """
    Move the file from one forder to another forder

    Parameters
    -------------
    src: str
        Relative path of the source forder
    dstp: str
        Relative path of the destination forder
    """

    for filepath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(filepath, filename), dstp)
