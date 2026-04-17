from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paper_reader.ai_summary import _build_codex_command


class PaperReaderAiSummaryTests(unittest.TestCase):
    def test_build_codex_command_uses_codex_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir) / "paper library"
            workdir.mkdir()
            output_path = Path(temp_dir) / "codex-last-message.md"
            with patch("src.paper_reader.ai_summary.shutil.which", return_value="/usr/bin/codex"):
                command = _build_codex_command(
                    workdir=workdir,
                    model="gpt-5.4",
                    output_path=output_path,
                )

        self.assertEqual(command[0], "codex")
        self.assertEqual(command[1], "exec")
        self.assertIn("--json", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertIn("--model", command)
        self.assertIn("gpt-5.4", command)
        self.assertIn(str(output_path), command)
        self.assertIn(str(workdir.resolve()), command)

    def test_build_codex_command_requires_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            output_path = workdir / "out.md"
            with patch("src.paper_reader.ai_summary.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "`codex` command is not available"):
                    _build_codex_command(
                        workdir=workdir,
                        model="gpt-5.4",
                        output_path=output_path,
                    )


if __name__ == "__main__":
    unittest.main()
