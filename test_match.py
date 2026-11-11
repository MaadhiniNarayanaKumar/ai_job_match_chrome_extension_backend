import unittest

from match import extract_skills_from_text, score_match


class ScoreMatchTests(unittest.TestCase):
    def test_broad_resume_still_covers_a_short_posting(self):
        result = score_match(
            "Python FastAPI Flask SQL Docker AWS git React pandas",
            "Python and FastAPI",
            [],
        )
        self.assertEqual(result["decision"], "APPLY")
        self.assertEqual(result["skill_match_score"], 100)
        self.assertEqual(result["missing_skills"], [])

    def test_missing_list_is_what_the_job_asked_for(self):
        result = score_match(
            "Python developer. FastAPI, Flask, SQL, Docker, AWS, git, and React.",
            "Python, FastAPI, PostgreSQL, Docker, AWS, React, and Kubernetes.",
            ["PostgreSQL"],
        )
        self.assertEqual(result["decision"], "APPLY")
        self.assertEqual(result["skill_match_score"], 86)
        self.assertEqual(result["missing_skills"], ["kubernetes"])
        self.assertIn("postgresql", result["matched_skills"])
        self.assertNotIn("flask", result["missing_skills"])

    def test_plain_english_is_not_a_skill(self):
        found = extract_skills_from_text("I will go to the store and call an api, then rest.")
        self.assertNotIn("go", found)
        self.assertNotIn("r", found)
        self.assertNotIn("api", found)

    def test_named_technologies_still_match(self):
        found = extract_skills_from_text("We use Golang and a REST API.")
        self.assertIn("golang", found)
        self.assertIn("rest api", found)

    def test_empty_posting_does_not_say_do_not_apply(self):
        result = score_match("Python and FastAPI", "Come join a friendly team.", [])
        self.assertEqual(result["decision"], "MAYBE APPLY")
        self.assertEqual(result["skill_match_score"], 0)
        self.assertIn("No skills", result["note"])


if __name__ == "__main__":
    unittest.main()
