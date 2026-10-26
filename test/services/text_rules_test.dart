import 'package:flutter_test/flutter_test.dart';
import 'package:neuralsafe/services/text_analysis_service.dart';
import 'package:neuralsafe/services/text_rules.dart';

// Edge-case regression tests for the text module's rule layer.
// 'B' = benign (idiom / negation / fiction): must be treated as Normal.
// 'D' = no rule fires: the model decides.
// 'R' = a rule must fire with the given class and at least the given score.
void main() {
  const normalProbs = [0.9, 0.04, 0.03, 0.03];

  group('TextRules.evaluate', () {
    const cases = <(String, String, String, int)>[
    ("He slapped me", 'R', 'Violence', 85),
    ("He is going to kill me tonight", 'R', 'Violence', 95),
    ("He locked me in the room and won't let me leave", 'R', 'Violence', 80),
    ("I'm scared, he hit me again and I'm bleeding", 'R', 'Violence', 85),
    ("He said if I tell anyone he will hurt my children", 'R', 'Violence', 70),
    ("He is choking me please help", 'R', 'Violence', 85),
    ("He is choking me", 'R', 'Violence', 85),
    ("He checks my phone and says he'll beat me if I talk to anyone", 'R', 'Violence', 70),
    ("He keeps threatening to take my kids away", 'R', 'Distress', 70),
    ("He screamed at me and broke my phone", 'R', 'Distress', 55),
    ("I'm afraid to go home tonight", 'R', 'Distress', 70),
    ("Please help me, call the police", 'R', 'Help', 85),
    ("Someone please come, I need help now", 'R', 'Help', 80),
    ("help", 'R', 'Help', 92),
    ("Help!", 'R', 'Help', 92),
    ("help me", 'R', 'Help', 92),
    ("please help me", 'R', 'Help', 92),
    ("bachao", 'R', 'Help', 92),
    ("Bachaaoooo", 'R', 'Help', 92),
    ("bachao mujhe bachao", 'R', 'Help', 88),
    ("Mujhe bachao", 'R', 'Help', 92),
    ("bachaaoooo mujhe bachaooo", 'R', 'Help', 88),
    ("مجھے بچاؤ", 'R', 'Help', 90),
    ("بچاؤ", 'R', 'Help', 92),
    ("madad", 'R', 'Help', 92),
    ("meri madad karo", 'R', 'Help', 92),
    ("hlp me pls he is hiting me", 'R', 'Violence', 85),
    ("Usne mujhe maara aur ghar se nikal diya", 'R', 'Violence', 90),
    ("Wo mujhe maarta hai", 'R', 'Violence', 90),
    ("Woh mujhe marta hai", 'R', 'Violence', 90),
    ("Usne mujhe mara", 'R', 'Violence', 90),
    ("Wo mujhe jaan se maar dega, mujhe bachao", 'R', 'Violence', 90),
    ("وہ مجھے مارتا ہے اور دھمکی دیتا ہے", 'R', 'Violence', 90),
    ("مجھے بچاؤ، وہ مجھے مار دے گا", 'R', 'Violence', 90),
    ("He ne mujhe thappar maara aur I am scared", 'R', 'Violence', 85),
    ("Mujhe bahut dar lag raha hai, wo phir ghusse mein hai", 'R', 'Distress', 55),
    ("I'm so scared, I can't stop shaking", 'R', 'Distress', 50),
    ("I can't take this anymore, I feel trapped", 'R', 'Distress', 66),
    ("I want to die", 'R', 'Distress', 92),
    ("I want to kill myself", 'R', 'Distress', 96),
    ("My neighbor's husband is beating her right now", 'R', 'Violence', 85),
    ("He has a knife and is pointing it at me", 'R', 'Violence', 90),
    ("He takes my salary and gives me 200 rupees a day", 'R', 'Distress', 62),
    ("He won't let me see my parents", 'R', 'Distress', 62),
    ("He says nobody would believe me if I told them", 'R', 'Distress', 62),
    ("Meri tankhwah wo le leta hai", 'R', 'Distress', 62),
    ("Please don't hurt me", 'R', 'Violence', 85),
    ("Please don't hit me", 'R', 'Violence', 85),
    ("He didn't stop hitting me", 'R', 'Violence', 70),
    ("He never stopped hitting me", 'R', 'Violence', 70),
    ("I can't breathe he is choking me", 'R', 'Violence', 85),
    ("He never hit me before, but now he threatened me", 'R', 'Distress', 72),
    ("He did not hurt me but he threatened to kill me", 'R', 'Violence', 95),
    ("My father used to hit me and it is getting worse", 'R', 'Violence', 70),
    ("I'm not scared of him anymore", 'B', '', 0),
    ("My brother is killing me with his jokes", 'B', '', 0),
    ("I'll kill this exam tomorrow, I studied all night", 'B', '', 0),
    ("I'll murder you if you eat my fries", 'B', '', 0),
    ("The movie had a scene where he hit her, it was so sad", 'B', '', 0),
    ("He never hit me, he would never hurt me", 'B', '', 0),
    ("My ex used to beat me but that was years ago", 'B', '', 0),
    ("My father used to hit me when I was a child", 'B', '', 0),
    ("That exam was a bloodbath", 'B', '', 0),
    ("I'm dying of laughter", 'B', '', 0),
    ("I'm dead tired", 'B', '', 0),
    ("My phone is dead", 'B', '', 0),
    ("Can you help me with my homework?", 'B', '', 0),
    ("I need help with my assignment", 'B', '', 0),
    ("I'm scared of spiders", 'B', '', 0),
    ("Hit me up later", 'B', '', 0),
    ("He beat me to it", 'B', '', 0),
    ("We need to beat the traffic", 'B', '', 0),
    ("Wo mujhe nahi maarta", 'B', '', 0),
    ("Mujhe ab dar nahi lagta", 'B', '', 0),
    ("میں اب ڈر نہیں", 'B', '', 0),
    ("وہ مجھے نہیں مارتا", 'B', '', 0),
    ("Dar lag raha hai horror movie dekh kar", 'B', '', 0),
    ("I am afraid I cannot attend the meeting", 'B', '', 0),
    ("Killed it at the interview", 'B', '', 0),
    ("We need to kill the lights", 'B', '', 0),
    ("I'm choking on my food", 'B', '', 0),
    ("paise bachao", 'D', '', 0),
    ("What time is dinner tonight?", 'D', '', 0),
    ("I'm going to the market, do you need anything?", 'D', '', 0),
    ("Happy birthday! See you at the party", 'D', '', 0),
    ("Aaj khana kya banana hai?", 'D', '', 0),
    ("ہمارا کھانا تیار ہے", 'D', '', 0),
    ("Please don't call me, I can't talk right now", 'D', '', 0),
    ("Ali hit a six in the match", 'B', '', 0),
    ("I am so happy today", 'D', '', 0),
    ("Can you help me?", 'R', 'Help', 85),
    ("I played football", 'B', '', 0),
    ("We played a cricket match today", 'B', '', 0),
    ("I am playing games on my phone", 'B', '', 0),
    ("He hit me with a cricket bat", 'R', 'Violence', 85),
    ("He beat me after the football match", 'R', 'Violence', 85),
    ];
    for (final c in cases) {
      test(c.$1, () {
        final v = TextRules.evaluate(c.$1);
        switch (c.$2) {
          case 'B':
            expect(v.benign, isTrue, reason: 'expected benign, got ${v.why}');
            expect(v.floor, 0);
            break;
          case 'D':
            expect(v.benign, isFalse);
            expect(v.floor, 0);
            break;
          default:
            expect(v.cls, c.$3);
            expect(v.floor, greaterThanOrEqualTo(c.$4.toDouble()));
        }
      });
    }
  });

  group('computeTextScore (model probabilities + rules)', () {
    test('joke with violent word stays Normal even if the model says Violence', () {
      final r = computeTextScore('My brother is killing me with his jokes',
          const [0.02, 0.03, 0.9, 0.05]);
      expect(r.predictedClass, 'Normal');
      expect(r.score, lessThanOrEqualTo(10));
    });

    test('negated fear stays Normal even if the model says Distress', () {
      final r = computeTextScore("I'm not scared of him anymore",
          const [0.05, 0.9, 0.03, 0.02]);
      expect(r.predictedClass, 'Normal');
      expect(r.score, lessThanOrEqualTo(10));
    });

    test('single-word plea is Help with a high score even if the model says Normal', () {
      for (final w in ['help', 'Help!', 'bachao', 'Bachaaoooo', 'madad']) {
        final r = computeTextScore(w, normalProbs);
        expect(r.predictedClass, 'Help', reason: w);
        expect(r.score, greaterThanOrEqualTo(88), reason: w);
      }
    });

    test('direct violence is flagged even if the model says Normal', () {
      final r = computeTextScore('He slapped me', normalProbs);
      expect(r.predictedClass, 'Violence');
      expect(r.score, greaterThanOrEqualTo(85));
    });

    test('harmless sentence scores near zero', () {
      final r = computeTextScore('What time is dinner tonight?', normalProbs);
      expect(r.predictedClass, 'Normal');
      expect(r.score, lessThan(10));
    });

    test('everyday sentence with a weak, uncertain Violence guess stays Normal', () {
      final r = computeTextScore('I played football', const [0.30, 0.15, 0.39, 0.16]);
      expect(r.predictedClass, 'Normal');
      expect(r.score, lessThanOrEqualTo(15));
    });

    test('uncertain model output with no rule is treated as no signal', () {
      final r = computeTextScore('blah blah blah', const [0.2, 0.25, 0.39, 0.16]);
      expect(r.predictedClass, 'Normal');
      expect(r.score, lessThanOrEqualTo(15));
    });

    test('model still decides when no rule fires', () {
      final r = computeTextScore('He makes me feel small', const [0.05, 0.9, 0.03, 0.02]);
      expect(r.predictedClass, 'Distress');
      expect(r.score, greaterThan(60));
    });
  });

  group('segments and normalization', () {
    test('splits sentences', () {
      final s = TextRules.splitSegments("Help! He hit me. I'm not scared of him anymore");
      expect(s.length, 3);
    });

    test('dangerous line hidden at the end of a long message is still found', () {
      final filler = List.filled(120, 'today was a normal day at work').join(' ');
      final text = '$filler and then he said he will kill me';
      final worst = TextRules.splitSegments(text)
          .map((seg) => TextRules.evaluate(seg).floor)
          .reduce((a, b) => a > b ? a : b);
      expect(worst, greaterThanOrEqualTo(95));
    });

    test('stretched letters are collapsed', () {
      expect(TextRules.normalize('Bachaaoooo'), ['bachaaoo']);
    });
  });
}
