"""Band calibration test.

Run:
  python -m scorer.ml.test_score_bands

This script uses three fixed essays that are expected (with the current scoring
calibration) to land in these overall score ranges:
- weak:   65-75
- average:76-89
- strong: 90-100

If the scorer logic changes (or the ML model artifact changes), these bands may
shift. The script prints diagnostics and exits non-zero if any case falls
outside its target band.
"""

from __future__ import annotations

from dataclasses import dataclass

from scorer.ml import scoring


@dataclass(frozen=True)
class BandCase:
    label: str
    min_total: float
    max_total: float
    essay: str


QUESTION = "Discuss the importance of education in modern society."

CASES: list[BandCase] = [
    BandCase(
        label="WEAK",
        min_total=65.0,
        max_total=75.0,
        essay=(
            "Education is important in modern society because it expands opportunity and equips people to adapt as technology, workplaces, and civic challenges change. Beyond literacy and numeracy, "
            "effective education develops critical thinking, communication, and the ability to learn continuously—skills that determine whether people can retrain, collaborate, and solve unfamiliar problems. "
            "One concrete example is media literacy: students who practice comparing sources and spotting weak reasoning are better prepared to resist misinformation and make informed choices about health, "
            "politics, and personal finance.\n\n"
            "Education also supports economic mobility and community stability. Credentials still influence hiring, but the deeper value is competence: reading technical documentation, interpreting basic data, "
            "and communicating clearly in teams. Consider how statistical understanding helps workers and citizens interpret charts, rates, and risk. When people can evaluate numbers and uncertainty, they are "
            "less likely to be misled by exaggerated headlines and more likely to support policies based on evidence. Consequently, education improves both productivity and the quality of public decision-making.\n\n"
            "A strong education system strengthens democracy as well. Learning history and civics helps people understand institutions and participate responsibly rather than reacting to slogans. It also teaches "
            "the habit of revising beliefs when evidence changes. This mindset matters during crises—public health, natural disasters, or economic shocks—when communities must coordinate and evaluate tradeoffs.\n\n"
            "Finally, education builds problem-solving habits that drive innovation. Students learn to test ideas, revise drafts, and work with others to design solutions. These habits translate into engineering, "
            "medicine, and entrepreneurship, but they also matter locally, such as improving traffic safety or creating community preparedness plans. Therefore, investing in accessible, high-quality education is one "
            "of the most practical ways to build individual success, social stability, and long-term resilience."
        ),
    ),
    BandCase(
        label="AVERAGE",
        min_total=76.0,
        max_total=89.0,
        essay=(
            "Education is important in modern society because it expands opportunity and equips people to adapt to rapid change. "
            "In a labor market shaped by automation, new tools, and shifting industries, the most valuable skill is not memorizing a fixed set of facts but learning how to learn. "
            "Good schooling develops literacy and numeracy, but it also builds critical thinking, communication, and the ability to evaluate evidence. For example, when students practice "
            "reading different sources on a controversial issue and comparing claims, they learn how to detect weak reasoning and misinformation. That habit matters when people make decisions "
            "about health advice, political messages, and financial products online.\n\n"
            "Education also supports economic mobility. Because credentials still influence hiring and wages, access to quality education can raise lifetime earnings and reduce poverty. "
            "However, the benefit is not only individual. As a result, communities with more educated residents often have higher employment, stronger tax bases, and more stable local businesses. "
            "In addition, education supports innovation: workers who can read technical documentation, interpret data, and collaborate across teams are more capable of adopting new methods. "
            "For instance, a technician who understands basic statistics and measurement can improve quality control, while a nurse with strong scientific literacy can apply updated clinical guidelines "
            "more reliably. Therefore, education increases both productivity and resilience during economic shocks.\n\n"
            "Education is also essential for democratic life. Modern societies depend on citizens who can understand how institutions work and who can disagree without abandoning facts. "
            "When learners study history and civics, they can recognize patterns like propaganda, scapegoating, and unfair policy design. For example, understanding how voting systems, budgets, "
            "and courts function helps people evaluate political promises and hold leaders accountable. As a result, education can reduce polarization by improving the quality of public discussion.\n\n"
            "Finally, education strengthens personal development and social connection. Schools can teach collaboration, time management, and persistence, which are needed in almost every career. "
            "They also provide structured opportunities to work with people from different backgrounds. For instance, group projects and classroom debates require students to listen, revise claims, "
            "and explain reasoning clearly. This is not always comfortable, but it builds empathy and problem-solving habits.\n\n"
            "In conclusion, education matters because it creates capable individuals and healthier communities. It prepares people for work, helps them evaluate information, and supports responsible civic "
            "participation. Therefore, investing in accessible, high-quality education is one of the most practical strategies for long-term economic strength, social stability, and opportunity."
        ),
    ),
    BandCase(
        label="STRONG",
        min_total=90.0,
        max_total=100.0,
        essay=(
            "Education is important in modern society because it expands opportunity and equips people to adapt to rapid change. "
            "In a labor market shaped by automation, new tools, and shifting industries, the most valuable skill is not memorizing a fixed set of facts but learning how to learn. "
            "Good schooling develops literacy and numeracy, but it also builds critical thinking, communication, and the ability to evaluate evidence. For example, when students practice "
            "reading different sources on a controversial issue and comparing claims, they learn how to detect weak reasoning and misinformation. That habit matters when people make decisions "
            "about health advice, political messages, and financial products online.\n\n"
            "Schooling also supports economic mobility. Because credentials still influence hiring and wages, access to quality instruction can raise lifetime earnings and reduce poverty. "
            "However, the benefit is not only individual. As a result, communities with more educated residents often have higher employment, stronger tax bases, and more stable local businesses. "
            "In addition, training supports innovation: workers who can read technical documentation, interpret data, and collaborate across teams are more capable of adopting new methods. "
            "For instance, a technician who understands basic statistics and measurement can improve quality control, while a nurse with strong scientific literacy can apply updated clinical guidelines "
            "more reliably. Therefore, learning increases both productivity and resilience during economic shocks.\n\n"
            "Education is also essential for democratic life. Modern societies depend on citizens who can understand how institutions work and who can disagree without abandoning facts. "
            "When learners study history and civics, they can recognize patterns like propaganda, scapegoating, and unfair policy design. For example, understanding how voting systems, budgets, "
            "and courts function helps people evaluate political promises and hold leaders accountable. As a result, education can reduce polarization by improving the quality of public discussion.\n\n"
            "Finally, education strengthens personal development and social connection. Schools can teach collaboration, time management, and persistence, which are needed in almost every career. "
            "They also provide structured opportunities to work with people from different backgrounds. For instance, group projects and classroom debates require students to listen, revise claims, "
            "and explain reasoning clearly. This is not always comfortable, but it builds empathy and problem-solving habits.\n\n"
            "In conclusion, education matters because it creates capable individuals and healthier communities. It prepares people for work, helps them evaluate information, and supports responsible civic "
            "participation. Therefore, investing in accessible, high-quality education is one of the most practical strategies for long-term economic strength, social stability, and opportunity.\n\n"
            "Another overlooked benefit is metacognition—learning how to monitor understanding and adjust strategies. Students who set goals, reflect on feedback, and revise plans develop habits that transfer "
            "to new domains. A design project, for instance, forces learners to define constraints, test prototypes, and document results. These routines mirror real-world problem-solving in fields ranging from "
            "software to public policy. It also encourages curiosity and ethical reflection, which matter when new technologies reshape daily life."
        ),
    ),
]


def main() -> int:
    print("QUESTION:", QUESTION)
    print("RUBRIC: content=50, grammar=30, formatting=20")
    print("LIMITS: min_words=250, max_words=1000")
    print("-" * 92)

    failed = 0

    for case in CASES:
        total, grammar, content, formatting, _ = scoring.total_score(QUESTION, case.essay)
        raw_ml = scoring.predict_ml_score(QUESTION, case.essay)
        stretched_ml = max(0.0, min(100.0, (raw_ml - 50.0) * 1.5 + 50.0))
        rep_quality = scoring._repetition_score(case.essay)
        word_count = scoring._count_words(case.essay)

        ok = case.min_total <= total <= case.max_total
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1

        print(f"{case.label}: {status} (target {case.min_total:.0f}-{case.max_total:.0f})")
        print(f"  words={word_count} repetition_quality={rep_quality:.3f}")
        print(f"  ml_raw={raw_ml:.2f} ml_stretched={stretched_ml:.2f}")
        print(
            f"  content={content:.2f} grammar={grammar:.2f} formatting={formatting:.2f} TOTAL={total:.2f}"
        )
        print("-" * 92)

    if failed:
        print(f"Band test failed: {failed} case(s) out of range")
        return 1

    print("Band test passed: all cases in-range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
