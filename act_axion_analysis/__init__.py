# Borrowed from the __init__.py file from socs on 12/10/2024 by ZBH
# https://github.com/simonsobs/socs/blob/main/socs/__init__.py

# Define the variable '__version__':
# This has the closest behavior to versioneer that I could find
# https://github.com/maresb/hatch-vcs-footgun-example
try:
    # If setuptools_scm is installed (e.g. in a development environment with
    # an editable install), then use it to determine the version dynamically.
    from setuptools_scm import get_version

    # This will fail with LookupError if the package is not installed in
    # editable mode or if Git is not installed.
    __version__ = get_version(root="..", relative_to=__file__, version_scheme="no-guess-dev")
except (ImportError, LookupError):
    # As a fallback, use the version that is hard-coded in the file.
    try:
        from act_axion_analysis._version import __version__  # noqa: F401
    except ModuleNotFoundError:
        # The user is probably trying to run this without having installed
        # the package, so complain.
        raise RuntimeError(
            "act_axion_analysis is not correctly installed. "
            "Please install it with pip."
        )