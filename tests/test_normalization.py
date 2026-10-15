import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from training.common.config import DataConfig, NormalizationConfig
from training.common.robots import RobotSchema
from training.model.normalization import (
    PretrainedNormalization, ResolvedNormalization,
    resolve_normalization, validate_normalization,
)


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.robot = RobotSchema(camera_keys=(), state_components=(("joint", 2),), action_components=(("joint", 2),))
        self.cfg = DataConfig(robot=self.robot, normalization=NormalizationConfig(
            explicit_mode="QUANTILES", stats_source="checkpoint"))
        self.stats = {column: {"q01": [0., 0.], "q99": [1., 1.]}
                      for column in ("observation.state", "action")}
        self.pretrained = PretrainedNormalization(
            mode={"STATE": "QUANTILES", "ACTION": "QUANTILES"},
            stats=self.stats, action_relative=False, action_relative_exclude=[],
            max_state_dim=2, max_action_dim=2, revision=None)
        self.model = SimpleNamespace(norm_tag=None)

    def resolve(self):
        with patch('training.model.normalization.inspect_pretrained_normalization', return_value=self.pretrained):
            return resolve_normalization(self.cfg, 'pi05', self.model, None)

    def validate(self, resolved=None, policy='pi05'):
        validate_normalization(resolved or self.resolve(), self.cfg, self.robot, policy, self.model)

    def test_valid_checkpoint_stats(self):
        self.validate()

    def test_checkpoint_stats_require_mode(self):
        self.cfg.normalization.explicit_mode = None
        with self.assertRaisesRegex(ValueError, 'require a resolved'):
            self.validate()

    def test_checkpoint_stats_require_matching_mode(self):
        self.cfg.normalization.explicit_mode = 'MEAN_STD'
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            self.validate()

    def test_missing_feature_or_stat_key(self):
        for source in ('checkpoint', 'explicit_file'):
            for feature in ('observation.state', 'action'):
                for missing in ('feature', 'q01', 'q99'):
                    with self.subTest(source=source, feature=feature, missing=missing):
                        stats = copy.deepcopy(self.stats)
                        if missing == 'feature':
                            del stats[feature]
                        else:
                            del stats[feature][missing]
                        self.cfg.normalization.stats_source = source
                        with self.assertRaisesRegex(ValueError, 'Missing normalization|require'):
                            self.validate(ResolvedNormalization(self.pretrained.mode, stats, self.pretrained))

    def test_checks_every_required_vector(self):
        for value in ([1.], [[1.], [2.]], [float('nan'), 1.], '12'):
            with self.subTest(value=value):
                stats = copy.deepcopy(self.stats)
                stats['action']['q99'] = value
                with self.assertRaisesRegex(ValueError, 'finite numeric vector'):
                    self.validate(ResolvedNormalization(self.pretrained.mode, stats, self.pretrained))

    def test_explicit_file_cannot_fall_back_to_dataset_stats(self):
        self.cfg.normalization.stats_source = 'explicit_file'
        for stats in ({}, {'action': {'mean': [0., 0.], 'std': [1., 1.]}}):
            with self.assertRaises(ValueError):
                self.validate(ResolvedNormalization(self.pretrained.mode, stats, None))

    def test_identity_needs_no_stats(self):
        self.cfg.normalization.stats_source = 'explicit_file'
        self.validate(ResolvedNormalization({'STATE': 'IDENTITY', 'ACTION': 'IDENTITY'}, {}, None))

    def test_pi05_stats_source_checks_capacity_and_action_semantics(self):
        for attr, value, message in (
            ('max_state_dim', 1, 'max_state_dim'),
            ('max_action_dim', 1, 'max_action_dim'),
            ('action_relative', True, 'action semantics'),
        ):
            with self.subTest(attr=attr):
                original = getattr(self.pretrained, attr)
                setattr(self.pretrained, attr, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                setattr(self.pretrained, attr, original)
        self.pretrained.action_relative = True
        self.pretrained.action_relative_exclude = ['gripper']
        self.cfg.action_representation = 'delta'
        with self.assertRaisesRegex(ValueError, 'exclude list'):
            self.validate()

    def test_molmoact2_does_not_use_pi05_action_semantics(self):
        self.cfg.action_representation = 'delta'
        self.validate(policy='molmoact2')

    def test_default_dataset_path_unchanged(self):
        self.cfg.normalization = NormalizationConfig()
        resolved = resolve_normalization(self.cfg, 'act', self.model, None)
        self.assertIsNone(resolved.mode)
        self.assertIsNone(resolved.stats)
        self.validate(resolved, policy='act')

    def test_dataset_mean_std_allows_precomputed_stats(self):
        self.cfg.normalization = NormalizationConfig(explicit_mode='MEAN_STD')
        self.validate(ResolvedNormalization({'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}, None, None))

    def test_dataset_min_max_already_rejected(self):
        self.cfg.normalization = NormalizationConfig(explicit_mode='MIN_MAX')
        with self.assertRaisesRegex(ValueError, 'no dataset-stats computation'):
            resolve_normalization(self.cfg, 'act', self.model, None)


if __name__ == '__main__':
    unittest.main()
