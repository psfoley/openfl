import hashlib
import os
import io
import zipfile

def get_src_fpaths(dir, sort=True):
    """Get a list of file paths from a folder of Python source code.

    Parameters
    ----------
    dir : src
        The path of the folder.
    sort : bool
        If we should sort the file paths.

    Returns
    -------
    list
        A list of string of file path.
    """
    filtered_ext = ['pyc']
    fpaths = []

    for dname, _, fnames in os.walk(dir):
        for fname in fnames:
            filtered = False
            for ext in filtered_ext:
                if fname.endswith(ext):
                    filtered = True
                    break
            
            if not filtered:
                fpath = os.path.join(dname, fname)
                fpaths.append(fpath)
    if sorted:
        fpaths = sorted(fpaths)
    return fpaths


def get_sha256_hash_from_files(fpaths):
    """ Calculate the sha256 checksum of a list of files.

    Parameters
    ----------
    fpath : list
        A list of string of file path.
    
    Returns
    -------
    str
        A hash string.
    """
    hasher = hashlib.sha256()
    for fpath in fpaths:
        with open(fpath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
    hash = hasher.hexdigest()
    return hash


def get_sha256_digest(bytes):
    """ Get sha256 digest from binary bytes.
    Parameters
    ----------
    bytes : bytes
        Binary byte string.

    Returns
    -------
    str
        Sha256 digest in string.
    """
    hasher = hashlib.sha256()
    hasher.update(bytes)
    hash = hasher.hexdigest()
    return hash

def check_bytes_sha256(bytes, ref_digest):
    """ Check the integrity of a binary byte string with sha256 digest.

    Parameters
    ----------
    bytes : bytes
        Binary byte string.
    ref_digest : str
        A reference sha256 digest.
    
    Returns
    -------
    bool
        True if the integrity is verified.
    """
    digest = get_sha256_digest(bytes)
    if digest == ref_digest:
        return True
    else:
        return False

def get_code_hash(dir):
    """ Get the hash string of a folder of Python source code.
    We traverse all files except *.pyc, sort the files and calcualte sha256.

    Parameters
    ----------
    dir : str
        The folder path of the code.

    Returns
    -------
    str
        A hash string.
    """
    fpaths = get_src_fpaths(dir, sort=True)
    hash = get_sha256_hash_from_files(fpaths)
    return hash


def zip_files(dir, compression = zipfile.ZIP_DEFLATED):
    """Zip a folder of files.

    Parameters
    ----------
    dir : str
        The folder containing subfolders and files.
    compression : int
        The compression lever from 0(fast) to 9(small). ZIP_DEFLATED==6 by default.

    Returns
    -------
    bytes
        The byte stream of the zip file.
    """
    fpaths = get_src_fpaths(dir, sort=True)
    zip_file = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(zip_file, 'w', compression=compression) as zip: 
        for fpath in fpaths:
            zip.write(fpath)

    zip_file.seek(0)
    ret = zip_file.read()
    zip_file.close()
    return ret


def unzip_files(zip_stream, target_dir, compression = zipfile.ZIP_DEFLATED):
    """Unzip files for a byte stream.

    Parameters
    ----------
    zip_stream : bytes
        The byte stream of a zip file.
    target_dir : str
        The target directory to store the unzipped files.
    compression : int
        The compression lever from 0(fast) to 9(small). ZIP_DEFLATED==6 by default.
    """
    zip_file = io.BytesIO(zip_stream)
    zip_file.seek(0)

    with zipfile.ZipFile(zip_file, 'r', compression=compression) as zip:
        # zip.printdir()
        zip.extractall(path=target_dir)
    zip_file.close()

if __name__ == "__main__":
    dir = "mnist_cnn_keras"
    zip_stream = zip_files(dir)
    hash = get_sha256_digest(zip_stream)

    print("We have got a zip stream in size of %d bytes with sha256 digest %s." % (len(zip_stream), hash))
    if check_bytes_sha256(zip_stream, hash):
        print("The integrity is verified.")
        target_dir = os.path.join(os.path.expanduser('~'), "extracted_code")
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        unzip_files(zip_stream, target_dir)
    else:
        print("We don't trust the modified copy.")
