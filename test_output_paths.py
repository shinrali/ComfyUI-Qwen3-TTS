import unittest

from output_paths import output_prefix_parts


class OutputPathTests(unittest.TestCase):
    def test_unicode(self):
        prefix = "pixion_voice/\u516b\u5b57\u89c6\u9891_\u5e74\u8f7b\u5973\u58f0_VOICE_T04"
        self.assertEqual(output_prefix_parts(prefix), prefix.split("/"))
        self.assertEqual(output_prefix_parts("voices/Caf\u00e9 voice"), ["voices", "Caf\u00e9 voice"])

    def test_backslash(self):
        self.assertEqual(output_prefix_parts("voices\\audio"), ["voices", "audio"])

    def test_unsafe(self):
        for prefix in ("", "/audio", "../audio", "voices/../audio", "C:\\audio", "//host/audio",
                       "voices//audio", "voices/audio/", "voices/.", "voice.", "voice ",
                       "a:b", "a?b", "a*b", 'a"b', "a<b", "a>b", "a|b", "a\x00b",
                       "a\nb", "CON", "con.wav", "LPT1", "COM\u00b9", "NUL .wav",
                       "a" * 129):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                output_prefix_parts(prefix)


if __name__ == "__main__":
    unittest.main()
