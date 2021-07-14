import pathlib
from setuptools import setup
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()setup(
  name="muldichinese",
  version="0.0.1",
  description="",
  long_description=README,
  long_description_content_type="text/pdf",
  author="",
  author_email="",
  license="GNU",
  packages=["muldichinese"],
  zip_safe=False
)
