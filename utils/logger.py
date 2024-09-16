"""Custom logger module"""
import logging
from colorama import init, Fore

# Initialize Colorama to support colored terminal output on Windows
init(autoreset=True)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a color mapping for different log levels
log_colors = {
    "DEBUG": Fore.WHITE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED,
}


# Create a custom log handler with colored output
class ColoredHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            message = self.format(record)
            level_color = log_colors[record.levelname]
            print(level_color + message)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


# Add the colored handler to the logger
colored_handler = ColoredHandler()
colored_handler.setLevel(logging.DEBUG)
colored_handler.setFormatter(formatter)
logger.addHandler(colored_handler)
