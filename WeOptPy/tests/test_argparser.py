# encoding=utf8

"""Argument parser test case module."""

from unittest import TestCase

from WeOptPy.util.argparser import (
	make_arg_parser,
	get_args,
	get_dict_args
)


class ArgParserTestCase(TestCase):
	def setUp(self):
		self.parser = make_arg_parser()

	def test_parser_fine(self):
		self.assertTrue(self.parser)

	def test_getArgs_fine(self):
		args = get_args(['-d', '10', '-no_fes', '100000000', '-a', 'SCA'])
		self.assertTrue(args)
		self.assertEqual(args.d, 10)
		self.assertEqual(args.no_fes, 100000000)
		self.assertEqual(args.algo, 'SCA')

	def test_getDictArgs_fine(self):
		args = get_dict_args(['-d', '10', '-no_fes', '100000000', '-a', 'SCA'])
		self.assertTrue(args)
		self.assertEqual(args['d'], 10)
		self.assertEqual(args['no_fes'], 100000000)
		self.assertEqual(args['algo'], 'SCA')
		self.assertEqual(args['seed'], [None])

	def test_getDictArgs_seed_fine(self):
		args = get_dict_args(['-d', '10', '-no_fes', '100000000', '-a', 'SCA', '-seed', '1'])
		self.assertTrue(args)
		self.assertEqual(args['d'], 10)
		self.assertEqual(args['no_fes'], 100000000)
		self.assertEqual(args['algo'], 'SCA')
		self.assertEqual(args['seed'], [1])

	def test_getDictArgs_seed_fine_two(self):
		args = get_dict_args(['-d', '10', '-no_fes', '100000000', '-a', 'SCA', '-seed', '1', '234', '231523'])
		self.assertTrue(args)
		self.assertEqual(args['d'], 10)
		self.assertEqual(args['no_fes'], 100000000)
		self.assertEqual(args['algo'], 'SCA')
		self.assertEqual(args['seed'], [1, 234, 231523])


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
