"""Exercise the real parser declarations and config merge without GPU imports."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class AdamCLIOverrideTests(unittest.TestCase):
    def test_partial_and_default_valued_overrides(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'training/train.py').read_text())
        main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        declarations = [n for n in main.body if isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Call) and n.value.args
                        and isinstance(n.value.args[0], ast.Constant)
                        and n.value.args[0].value in ('--adam-beta1', '--adam-beta2')]
        merge = next(n for n in main.body if isinstance(n, ast.Assign)
                     and any(ast.unparse(t) == 'run_cfg.train.adam_betas' for t in n.targets))
        self.assertEqual(len(declarations), 2)
        for argv, expected in (
            ([], (.85, .95)),
            (['--adam-beta1', '.8'], (.8, .95)),
            (['--adam-beta2', '.98'], (.85, .98)),
            (['--adam-beta1', '.9'], (.9, .95)),
            (['--adam-beta2', '.999'], (.85, .999)),
            (['--adam-beta1=.9', '--adam-beta2=.999'], (.9, .999)),
        ):
            with self.subTest(argv=argv):
                namespace = {'parser': argparse.ArgumentParser(),
                             'run_cfg': SimpleNamespace(train=SimpleNamespace(adam_betas=(.85, .95)))}
                exec(compile(ast.Module(body=declarations, type_ignores=[]), '<parser>', 'exec'), namespace)
                namespace['args'] = namespace['parser'].parse_args(argv)
                exec(compile(ast.Module(body=[merge], type_ignores=[]), '<config-merge>', 'exec'), namespace)
                self.assertEqual(namespace['run_cfg'].train.adam_betas, expected)
