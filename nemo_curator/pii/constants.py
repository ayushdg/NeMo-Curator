DEFAULT_LANGUAGE = "en"

SUPPORTED_ENTITIES = [
    "ADDRESS",
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "IP_ADDRESS",
    "LOCATION",
    "PERSON",
    "URL",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "PHONE_NUMBER",
]

DEFAULT_MAX_DOC_SIZE = 2000000

__all__ = ["DEFAULT_LANGUAGE", "SUPPORTED_ENTITIES", "DEFAULT_MAX_DOC_SIZE"]