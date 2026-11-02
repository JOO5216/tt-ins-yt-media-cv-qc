import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_media_specs as specs
import face_size_check as face


class QualityChecks(unittest.TestCase):
    def test_spec_boundaries(self):
        for fps, duration, passed in [('24', '5', True), ('24', '60', True),
                                      ('24000/1001', '5', False), ('24', '60.01', False)]:
            with self.subTest(fps=fps, duration=duration):
                data = {'format': {'format_name': 'mp4', 'duration': duration},
                        'streams': [{'codec_type': 'video', 'codec_name': 'h264',
                                     'width': 720, 'height': 1280, 'avg_frame_rate': fps}]}
                with patch.object(specs.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(data))) as run:
                    self.assertEqual(specs.check_video(Path('test.mp4'), 'ffprobe')[0], passed)
                    self.assertEqual(run.call_args.kwargs['timeout'], 60)

    def test_spec_report_survives_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'a.mp4').touch(); (root/'b.mp4').touch()
            with patch('sys.argv', ['check_media_specs.py', temp]), \
                    patch.object(specs.shutil, 'which', return_value='ffprobe'), \
                    patch.object(specs, 'check_video', side_effect=[(True, 'valid', []), KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    specs.main()
            with next(root.glob('媒体规格检查_*.csv')).open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['结论'], '通过')

    def test_face_verdicts(self):
        for statuses, expected in [(['合格']*3, '合格'),
                                   (['合格', '未检出', '合格'], '需复核'),
                                   (['合格', '不足128px', '未检出'], '不合格')]:
            with self.subTest(statuses=statuses), patch.object(face, 'get_duration', return_value=10), \
                    patch.object(face, 'extract_frame'), patch.object(face, 'detect_largest_face',
                    side_effect=[(128, 128, .9, status) for status in statuses]):
                self.assertEqual(face.check_video(None, None, Path('x.mp4'), 'x', 'x', Path('.'))[2], expected)

    def test_face_external_commands_have_timeouts(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(face.subprocess, 'run',
                side_effect=subprocess.TimeoutExpired('tool', 60)) as run:
            operations = [(lambda: face.get_duration(Path('x.mp4')), 60),
                          (lambda: face.extract_frame(Path('x.mp4'), 1, Path(temp)/'x.jpg'), 60),
                          (lambda: face.download_tiktok('yt-dlp', 'https://www.tiktok.com/test', Path(temp)/'download'), 300)]
            for operation, seconds in operations:
                with self.assertRaises(subprocess.TimeoutExpired):
                    operation()
                self.assertEqual(run.call_args.kwargs['timeout'], seconds)

    def test_invalid_metadata_is_rejected(self):
        for field, values in [('duration', ['0', '-1', 'nan', 'inf', 'N/A']),
                              ('avg_frame_rate', ['0', '-1', '0/0', 'N/A'])]:
            for value in values:
                with self.subTest(field=field, value=value):
                    data = {'format': {'format_name': 'mp4', 'duration': '5'},
                            'streams': [{'codec_type': 'video', 'codec_name': 'h264',
                                         'width': 1280, 'height': 720, 'avg_frame_rate': '24'}]}
                    (data['format'] if field == 'duration' else data['streams'][0])[field] = value
                    with patch.object(specs.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(data))):
                        self.assertFalse(specs.check_video(Path('test.mp4'), 'ffprobe')[0])

    def test_probe_timeout_is_reported(self):
        with patch.object(specs.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffprobe', 60)):
            self.assertFalse(specs.check_video(Path('test.mp4'), 'ffprobe')[0])

    def test_face_rejects_nonfinite_duration(self):
        for value in ['nan', 'inf', '0', '-1']:
            with self.subTest(value=value), patch.object(face.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps({'format': {'duration': value}}))):
                with self.assertRaises(RuntimeError):
                    face.get_duration(Path('test.mp4'))

    def run_face_batch(self, root, second_result):
        for name in ['a.mp4', 'b.mp4']:
            (root / name).touch()
        calls = 0

        def check(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                reports = list(root.parent.glob('*_人脸尺寸检查_*/人脸尺寸检查.csv'))
                with reports[0].open(encoding='utf-8-sig', newline='') as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 1)
                raise second_result
            return 'a.mp4', 5, '合格', [('前', .5, 140, 140, .9, '合格', root/'frame.jpg')]

        fake_cv = SimpleNamespace(FaceDetectorYN=SimpleNamespace(create=lambda *a: None))
        with patch('sys.argv', ['face_size_check.py', str(root)]), \
                patch.object(face, 'require_tools', return_value=fake_cv), \
                patch.object(face, 'ensure_model'), patch.object(face, 'check_video', side_effect=check):
            return face.main()

    def test_face_flushes_before_next_video_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'input'; root.mkdir()
            self.assertEqual(self.run_face_batch(root, RuntimeError('bad video')), 2)
            with next(Path(temp).glob('*_人脸尺寸检查_*/人脸尺寸检查.csv')).open(encoding='utf-8-sig') as f:
                self.assertEqual(list(csv.DictReader(f))[-1]['视频结论'], '处理失败')

    def test_face_keeps_completed_rows_on_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'input'; root.mkdir()
            with self.assertRaises(KeyboardInterrupt):
                self.run_face_batch(root, KeyboardInterrupt())


if __name__ == '__main__':
    unittest.main()
