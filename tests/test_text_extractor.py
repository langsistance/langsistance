"""Tests for USPTO xmlarchive tar extraction (text_extractor)."""
import io
import tarfile
import unittest

from sources.long_task.text_extractor import extract_text_from_xmlarchive_tar


def _tar_with(members: dict[str, str]) -> bytes:
    """Build a tar archive from {member_name: text_content}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestGetDownloadUrlStrictMode(unittest.TestCase):
    DOC = {
        "downloadOptionBag": [
            {"mimeTypeIdentifier": "XML", "downloadUrl": "https://x/x.xmlarchive"},
            {"mimeTypeIdentifier": "PDF", "downloadUrl": "https://x/x.pdf"},
        ]
    }

    def test_strict_mode_returns_empty_when_order_unmatched(self):
        from sources.long_task.text_extractor import get_download_url_from_doc
        self.assertEqual(
            get_download_url_from_doc(
                self.DOC,
                mime_order=("MS_WORD",),
                fallback_to_any=False,
            ),
            "",
        )

    def test_default_falls_back_to_any(self):
        from sources.long_task.text_extractor import get_download_url_from_doc
        self.assertTrue(
            get_download_url_from_doc(self.DOC, mime_order=("MS_WORD",))
        )


class TestXmlarchiveTarExtraction(unittest.TestCase):
    def test_extracts_clm_xml_member(self):
        content = "<claims><claim num='00001'><claim-text>claim one</claim-text></claim></claims>"
        tar = _tar_with({"12098926/GW/12098926.CLM.xml": content})
        self.assertEqual(extract_text_from_xmlarchive_tar(tar), content)

    def test_prefers_spec_xml_when_both_present(self):
        spec = "<specification>spec text</specification>"
        clm = "<claims>claims text</claims>"
        tar = _tar_with({
            "app/doc/app.SPEC.XML": spec,
            "app/doc/app.CLM.xml": clm,
        })
        self.assertEqual(extract_text_from_xmlarchive_tar(tar), spec)

    def test_falls_back_to_any_xml_member(self):
        content = "<other>some meaningful document text here</other>"
        tar = _tar_with({"app/doc/app.OTHER.xml": content})
        self.assertEqual(extract_text_from_xmlarchive_tar(tar), content)

    def test_skips_zip_and_svg_members(self):
        tar = _tar_with({
            "app/doc/app.SPEC_svg.zip": "zip bytes",
            "app/doc/drawing.svg": "<svg/>",
        })
        self.assertIsNone(extract_text_from_xmlarchive_tar(tar))

    def test_returns_none_when_no_xml_member(self):
        tar = _tar_with({"app/doc/only.zip": "zip bytes"})
        self.assertIsNone(extract_text_from_xmlarchive_tar(tar))


if __name__ == "__main__":
    unittest.main()
