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
