class AWSCpuType:
    ORDER = ["C4", "M4", "R4"]
    """
    aws cpu type
    """
    C4L = "C4"
    M4L = "M4"
    R4L = "R4"
    C4XL = "C4"
    M4XL = "M4"
    R4XL = "R4"
    C4XXL = "C4"
    M4XXL = "M4"
    R4XXL = "R4"


class AWSCoreCount:
    ORDER = [2.0, 4.0, 8.0]
    """
    aws core count
    """
    C4L = 2.0
    M4L = 2.0
    R4L = 2.0
    C4XL = 4.0
    M4XL = 4.0
    R4XL = 4.0
    C4XXL = 8.0
    M4XXL = 8.0
    R4XXL = 8.0


class AWSCpuSpeedPerCore:
    ORDER = [2.3, 2.4, 2.9]
    """
    aws CPU Speed per core
    """
    C4L = 2.9
    M4L = 2.4
    R4L = 2.3
    C4XL = 2.9
    M4XL = 2.4
    R4XL = 2.3
    C4XXL = 2.9
    M4XXL = 2.4
    R4XXL = 2.3


class AWSRamPerCore:
    ORDER = [2.0, 4.0, 8.0]
    """
    aws RAM per core
    """
    C4L = 2.0
    M4L = 4.0
    R4L = 8.0
    C4XL = 2.0
    M4XL = 4.0
    R4XL = 8.0
    C4XXL = 2.0
    M4XXL = 4.0
    R4XXL = 8.0


class AWSDiskType:
    ORDER = ["EBS"]
    """
    aws disk type
    """
    C4L = "EBS"
    M4L = "EBS"
    R4L = "EBS"
    C4XL = "EBS"
    M4XL = "EBS"
    R4XL = "EBS"
    C4XXL = "EBS"
    M4XXL = "EBS"
    R4XXL = "EBS"


class AWSDiskSpeed:
    ORDER = [500.0, 750.0, 1000.0]
    """
    aws disk speed
    """
    C4L = 500.0
    M4L = 500.0
    R4L = 500.0  # hacked
    C4XL = 750.0
    M4XL = 750.0
    R4XL = 750.0  # hacked
    C4XXL = 1000.0
    M4XXL = 1000.0
    R4XXL = 1000.0  # hacked


class AWSNetworkCapacity:
    ORDER = ["Moderate", "High", "UpToTen"]
    """
    aws network capacity
    """
    C4L = "Moderate"
    M4L = "Moderate"
    R4L = "UpToTen"
    C4XL = "High"
    M4XL = "High"
    R4XL = "UpToTen"
    C4XXL = "High"
    M4XXL = "High"
    R4XXL = "UpToTen"

