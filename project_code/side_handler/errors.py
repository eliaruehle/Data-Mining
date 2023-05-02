class NoSuchPathOrCSV(Exception):
    """
    Special Exception if a requestes path or csv is not present.
    """

    pass


class ClusterFormatError(Exception):
    """
    Special Exception if there is an Error if we read in content from an .csv file.
    """

    pass


class NoSuchClusterMethodError(Exception):
    """
    Special Exception if you want to establish a cluster strategy which isn't configured.
    """

    pass
