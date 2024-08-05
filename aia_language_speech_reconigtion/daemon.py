from argparse import ONE_OR_MORE, ArgumentParser
from . import __version__
from .speechReconigtion import startSpeechReconigtion

def run():
    """
    entry point
    """
    parser = ArgumentParser(prog="daemon", description="Speech Reconigtion")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    #parser.add_argument(dest="users", nargs=ONE_OR_MORE, type="User", help="your name")
    #args = parser.parse_args()
    startSpeechReconigtion()