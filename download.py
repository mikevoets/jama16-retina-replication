########################################################################
#
# Functions for downloading and extracting data-files from the internet.
#
# Implemented in Python 3.5
#
########################################################################

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _extract(file_path, extract_dir):
    """
    Extracts the given file path assuming that it is a tarball or zip-file.
    """
    if ".zip" in file_path:
        # Unpack the zip-file.
        zipfile.ZipFile(file=file_path, mode="r").extractall(extract_dir)
    elif ".tar.gz" in file_path or ".tgz" in file_path:
        # Unpack the tar-ball.
        tarfile.open(name=file_path, mode="r:gz").extractall(extract_dir)


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.

    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.uit.no/data/eyepacs.tar.gz"

    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/eyepacs/"

    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(
            url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        _extract(file_path=file_path, extract_dir=download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


########################################################################


def maybe_extract(file_path, extract_dir):
    """
    Extract the data if it doesn't already exist.

    :param filename:
        Filename of zip/tarball to extract.
        Example: "trainData.zip.001"

    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/eyepacs/"

    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = file_path.split('/')[-1].split('.')[0]
    extract_path = os.path.join(extract_dir, filename)

    # Check if the extract dir already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to and extract it now.
    if not os.path.exists(extract_path):
        # Check if the extract directory exists, otherwise create it.
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        _extract(file_path=file_path, extract_dir=extract_dir)

        print("Done.")
    else:
        print("Data has apparently already been unpacked.")


########################################################################
