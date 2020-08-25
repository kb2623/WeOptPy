# encoding=utf8

"""Unit tests configuration file."""

import logging


def pytest_configure(config):
	"""Disable verbose output when running tests."""
	logging.basicConfig(level=logging.DEBUG)


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
