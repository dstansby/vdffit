from pathlib import Path

import sunpy

data_dir = Path(sunpy.config.get('downloads', 'download_dir'))
